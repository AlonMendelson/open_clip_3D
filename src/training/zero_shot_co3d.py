import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

from open_clip import tokenize
from .co3d_zeroshot_data import co3d_classnames, co3d_template, co3d_loss_template

from torchmetrics import ConfusionMatrix


def zero_shot_classifier(model, classnames, templates, args):
    bins = int(math.floor(180/args.granularity))
    with torch.no_grad():
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
        if args.distributed and not args.horovod:
            class_embeddings = model.module.encode_text(texts, views)
        else:
            class_embeddings = model.encode_text(texts, views)
        zeroshot_weights = class_embeddings
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    with torch.no_grad():
        #confmat = ConfusionMatrix(num_classes=classifier.shape[0])
        top1, top3, n = 0., 0., 0.
        for images, targets in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            target = targets.to(args.device)
            target = torch.argmax(target,dim=1)

            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ torch.transpose(classifier,0,1)

            # measure accuracy
            acc1, acc3 = accuracy(logits, target, topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)
            #confmat.update(torch.argmax(logits,dim=1).detach().cpu(),target.detach().cpu())
    #confusion_tensor = confmat.compute()
    top1 = (top1 / n)
    top3 = (top3 / n)
    return top1, top3#, confusion_tensor


def zero_shot_eval_co3d(model, data, epoch, name, args):
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting' + name)

    logging.info('Building classifier')

    
    classifier = zero_shot_classifier(model, co3d_classnames, co3d_template, args)

    logging.info('Using classifier')
    results = {}
    #top1, top3, confmat = run(model, classifier, data.dataloader, args)
    top1, top3 = run(model, classifier, data.dataloader, args)
    results[name + "-top1"] = top1
    results[name + "-top3"] = top3

    logging.info('Finished '+ name)

    return results#, confmat
