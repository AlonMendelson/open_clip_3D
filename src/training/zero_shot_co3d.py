import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import tokenize
from .co3d_zeroshot_data import co3d_classnames, co3d_template


def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        new_classnames = []
        zeroshot_weights = []
        for classname in tqdm(classnames):
            new_classnames += [template(classname) for template in templates]  # format with class
        texts = tokenize(new_classnames).to(args.device)  # tokenize
        if args.distributed and not args.horovod:
            class_embeddings = model.module.encode_text(texts)
        else:
            class_embeddings = model.encode_text(texts)
        zeroshot_weights = class_embeddings
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    with torch.no_grad():
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

    top1 = (top1 / n)
    top3 = (top3 / n)
    return top1, top3


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
    top1, top3 = run(model, classifier, data.dataloader, args)
    results[name + "-top1"] = top1
    results[name + "-top3"] = top3

    logging.info('Finished '+ name)

    return results
