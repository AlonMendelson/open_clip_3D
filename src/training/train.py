import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, CEClipRegLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .zero_shot_co3d import zero_shot_eval_co3d
from open_clip import tokenize
from .co3d_zeroshot_data import co3d_classnames, co3d_template, co3d_loss_template

def create_classifier(model, classnames, templates, args,with_grad):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenize(texts).to(args.device)  # tokenize
            with autocast():
                if args.distributed and not args.horovod:
                    class_embeddings = model.module.encode_text(texts)
                else:
                    class_embeddings = model.encode_text(texts)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    zeroshot_weights = zeroshot_weights.view(-1,zeroshot_weights.shape[2])        
    return zeroshot_weights


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, reference_model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    bins = int(math.floor(180/args.granularity))

    model.train()
    loss = torch.nn.CrossEntropyLoss()
    #loss = CEClipRegLoss(reference_model,args)

    #data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    sampler = data["train"].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):

        #create classifier
        texts = []
        views = []
        for classname in co3d_classnames:
            texts_class = [template(classname) for template in co3d_loss_template for l in range(bins)]  # format with class
            texts_class = tokenize(texts_class).to(args.device)  # tokenize
            texts.append(texts_class)
            views.append(torch.arange(0,bins,1,dtype=texts_class.dtype,device=args.device))
        texts = torch.stack(texts, dim=0).to(args.device)
        views = torch.stack(views,dim=0).to(args.device)
        texts = texts.view(-1,texts.shape[2])
        views = views.view(-1,1)


        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, targets = batch
        images = images.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features,text_features, logit_scale = model(images,texts,view = views)
            logits = logit_scale * image_features @ text_features.t()
            #try using the gather they do
            total_loss = loss(logits,targets)
            #total_loss = loss(model,logits,targets)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    bins = int(math.floor(180/args.granularity))
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    co3d_in_val_metric, confmat_in = zero_shot_eval_co3d(model, data["val_in"], epoch, "val-in-co3d",args)
    metrics.update(co3d_in_val_metric)

    co3d_out_val_metric, confmat_out = zero_shot_eval_co3d(model, data["val_out"], epoch, "val-out-co3d",args)
    metrics.update(co3d_out_val_metric)

    if args.zeroshot_data:
        co3d_zeroshot_metric, confmat_zeroshot = zero_shot_eval_co3d(model, data["zeroshot"], epoch, "zeroshot-co3d",args)
        metrics.update(co3d_zeroshot_metric)


    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val_in' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val_in'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, targets = batch
                images = images.to(device=device, non_blocking=True)
                texts = []
                views = []
                for classname in co3d_classnames:
                    texts_class = [template(classname) for template in co3d_loss_template for l in range(bins)]  # format with class
                    texts_class = tokenize(texts_class).to(args.device)  # tokenize
                    texts.append(texts_class)
                    views.append(torch.arange(0,bins,1,dtype=texts_class.dtype,device=args.device))
                texts = torch.stack(texts, dim=0).to(args.device)
                views = torch.stack(views,dim=0).to(args.device)
                texts = texts.view(-1,texts.shape[2])
                views = views.view(-1,1)
                targets = targets.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts, views)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    logit_scale = logit_scale.mean()
                    logits = logit_scale * image_features @text_features.t()
                    #logits = F.softmax(logits,dim=1)
                    batch_size = images.shape[0]
                    total_loss = F.cross_entropy(logits, targets)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")


            loss = cumulative_loss / num_samples
            metrics.update(
                {"val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

        torch.save(confmat_in,
                    os.path.join(args.checkpoint_path, f"epoch_{epoch}_confmat_in.pt"))
        torch.save(confmat_out,
                    os.path.join(args.checkpoint_path, f"epoch_{epoch}_confmat_out.pt"))
        torch.save(confmat_zeroshot,
                    os.path.join(args.checkpoint_path, f"epoch_{epoch}_confmat_zeroshot.pt"))

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics