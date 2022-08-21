#!/bin/bash
torchrun --nproc_per_node 8 -m training.main --train-data ../../../../datasets/co3d   --val-data ../../../../datasets/co3d --zeroshot-data ../../../../datasets/co3d --imagenet-val ../../../../datasets/ilsvrc/val --model ViT-B-32  --dataset-type co3d --pretrained openai  --workers 4 --report-to tensorboard 
#python -m training.main --train-data ../../../../datasets/co3d   --val-data ../../../../datasets/co3d --zeroshot-data ../../../../datasets/co3d --imagenet-val ../../../../datasets/ilsvrc/val --model ViT-B-32  --dataset-type co3d --pretrained openai  --workers 4 --report-to tensorboard --lock-image
#torchrun --nproc_per_node 8 -m training.main  --val-data ../../../../datasets/co3d --zeroshot-data ../../../../datasets/co3d --imagenet-val ../../../../datasets/ilsvrc/val --model ViT-B-32 --pretrained openai  --dataset-type co3d  --workers 4 --wise logs/2022_08_07-13_10_54-model_ViT-B-32-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_22.pt --wise-alpha 0.7