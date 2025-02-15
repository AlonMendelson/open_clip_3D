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

from open_clip import ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval


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


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    #loss = ClipLoss(
    #    local_loss=args.local_loss,
    #    gather_with_grad=args.gather_with_grad,
    #    cache_labels=True,
    #    rank=args.rank,
    #    world_size=args.world_size,
    #    use_horovod=args.horovod)

    #####model for regression#####
    loss = torch.nn.MSELoss()
    ##############################

    #####model for classification#####
    #loss = torch.nn.CrossEntropyLoss()
    ##################################

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    #####model for classification#####
    #acc_m = AverageMeter()
    ##################################
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        ref_images, pred_images, texts, angles, path_ref, path_rot = batch
        ref_images = ref_images.to(device=device, non_blocking=True)
        pred_images = pred_images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        angles = angles.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            pred_angles = model(ref_images, pred_images, texts)
            total_loss = loss(angles, pred_angles)
        
        #####model for classification#####
        #prediction = torch.max(pred_angles.data, 1)[1]
        #truth = torch.max(angles.data, 1)[1]
        #correct = (prediction == truth).sum().item()/(angles.size(0))
        ##################################

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
        #with torch.no_grad():
        #    unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(ref_images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            #####model for classification#####
            #acc_m.update(correct,batch_size)
            ##################################
            #logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                #####model for classification#####
                #f"Accuracy: {acc_m.val:#.5g}({acc_m.avg:#.4g})"
                ##################################
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
             #   f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                #####model for classification#####
                #"accuracy": acc_m.val,
                ##################################
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                #"scale":  logit_scale_scalar,
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
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    #zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    #metrics.update(zero_shot_metrics)

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        #per_box_cumulative_loss = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        #per_box_num_samples = [0,0,0,0,0,0,0,0,0,0]
        #cumulative_correct = 0.0
        #all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                ref_images, pred_images, texts, angles, path_ref, path_rot = batch
                ref_images = ref_images.to(device=device, non_blocking=True)
                pred_images = pred_images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                angles = angles.to(device=device, non_blocking=True)

                with autocast():
                    #image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    #all_image_features.append(image_features.cpu())
                    #all_text_features.append(text_features.cpu())
                    #logit_scale = logit_scale.mean()
                    #logits_per_image = logit_scale * image_features @ text_features.t()
                    #logits_per_text = logits_per_image.t()
                    pred_angles = model(ref_images, pred_images, texts)

                    batch_size = ref_images.shape[0]
                    #####model for regression#####
                    total_loss = F.l1_loss(angles, pred_angles)
                    ##############################
                    #angles_np = angles.detach().cpu().numpy()
                    #pred_angles_np = pred_angles.detach().cpu().numpy()
                    #for l in range(angles_np.size):
                    #    box = int(angles_np[l][0]/0.1)
                    #    per_box_num_samples[box] += 1
                    #    per_box_cumulative_loss[box] += np.abs(angles_np[l][0]-pred_angles_np[l][0])
                    #####model for classification#####
                    #total_loss = F.cross_entropy(angles,pred_angles)
                    ##################################

                    #worst = 0
                    #ref = path_ref[0][worst]
                    #rot = path_rot[0][worst]
                    #ang = angles[worst].detach().cpu().item()
                    #pred = pred_angles[worst].detach().cpu().item()
                    #line = "the ref path is {path1}, the rot path is {path2}, the GT angle is {gt}, the predicted angle is {p} \n".format(path1 = ref, path2 = rot, gt = ang, p = pred)
                    #with open('examples_car1.txt','a') as f:
                    #    f.write(line)
                    #    f.close()
                #####model for classification#####
                #prediction = torch.max(pred_angles.data, 1)[1]
                #truth = torch.max(angles.data, 1)[1]
                #correct = (prediction == truth).sum().item()/batch_size
                #cumulative_correct += correct*batch_size
                ##################################
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")
                        #f"Accuracy: {cumulative_correct / num_samples:.6f}\t")

            #val_metrics = get_metrics(
            #    image_features=torch.cat(all_image_features),
            #    text_features=torch.cat(all_text_features),
            #    logit_scale=logit_scale.cpu(),
            #)
            #per_box_cumulative_loss = np.array(per_box_cumulative_loss)
            #per_box_num_samples = np.array(per_box_num_samples)
            #per_box_avg_loss_deg = np.divide(per_box_cumulative_loss,per_box_num_samples)
            
            #with open('avg_loss2.txt','w') as f:
            #    f.write("start\n")
            #    for t in range(per_box_avg_loss_deg.size):
            #        line = "box {box_num}: cummulative loss {c_loss}, num samples {n_sam}, avg {score}\n".format(box_num =t ,c_loss = per_box_cumulative_loss[t], n_sam = per_box_num_samples[t], score = per_box_avg_loss_deg[t])
            #        f.write(line)
            loss = cumulative_loss / num_samples
            #acc = cumulative_correct / num_samples
            metrics.update(
                {"val_loss": loss.item(),
                 #"val_acc": acc, 
                  "epoch": epoch, "num_samples": num_samples}
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
