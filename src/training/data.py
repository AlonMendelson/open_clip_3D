import ast
import json
import logging
import math
import os
import random
from dataclasses import dataclass
import pickle
import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from .co3d_zeroshot_data import co3d_classnames, co3d_template


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize

import training.co3d_dataset_arrange as co3d_dataset_arrange

import gzip

import training.co3d_types as co3d_types

from typing import List

from scipy.spatial.transform import Rotation as Rot


class Co3dDataset_New(Dataset):
    def __init__(self,transforms,dataset_root,annotations_root,mode,categories):
        logging.debug(f'Creating data from {dataset_root}.')
        self.dataset_root = dataset_root
        self.annotations_root = annotations_root
        self.transforms = transforms
        self.mode = mode
        if mode == "train":
            pickle_file_path = os.path.join(dataset_root,"co3d_train.pkl")
        elif mode == "val":
            pickle_file_path = os.path.join(dataset_root,"co3d_val.pkl")
        else:
            pickle_file_path = os.path.join(dataset_root,"co3d_leftout.pkl")

        self.all_samples = pd.read_pickle(pickle_file_path)
        self.classnames = []
        for classname in co3d_classnames:
            self.classnames += [template(classname) for template in co3d_template]  

    
    def __len__(self):
        return len(self.all_samples)

    def _load_16big_png_depth(self,depth_png):
        with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
        return depth

    def _load_depth(self,path, scale_adjustment):
        if not path.lower().endswith(".jpg.geometric.png"):
            raise ValueError('unsupported depth file name "%s"' % path)

        d = self._load_16big_png_depth(path) * scale_adjustment
        d[~np.isfinite(d)] = 0.0
        return d[None]  # fake feature channel

    def __getitem__(self, idx):

        sample = self.all_samples[idx]
        image_path = sample[0]
        depth_path = sample[1]
        scale = sample[2]
        mask_depth_path = sample[3]
        gt_pose = sample[4]
        sample_category = sample[5]

        image = self.transforms(Image.open(os.path.join(self.dataset_root,image_path)))
        #TODO: Crop depth
        depth = self._load_depth(os.path.join(self.dataset_root,depth_path),scale)

        r = Rot.from_matrix(gt_pose)
        angle = r.as_euler('yzx',degrees=True)[0]
        angle = abs(angle)
        qunatized_angle = int(math.floor(angle/18)*18)
        if qunatized_angle == 0:
            caption = "a photo of a " + sample_category +"."
        else:
            caption = "a photo of a " + sample_category + " rotated by " + str(qunatized_angle) + " degrees."
        
        target = self.classnames.index(caption)
        text = tokenize([caption])[0]


        return image, text, target

class Co3dDataset_CE(Dataset):
    def __init__(self,transforms,dataset_root,annotations_root,mode,categories):
        logging.debug(f'Creating data from {dataset_root}.')
        self.dataset_root = dataset_root
        self.annotations_root = annotations_root
        self.transforms = transforms
        self.mode = mode
        if mode=="train":
            samples_filename = "co3d_train.pkl"
        elif mode=="val_in":
            samples_filename = "co3d_val_in.pkl"
        elif mode=="val_out":
            samples_filename = "co3d_val_out.pkl"
        else:
            samples_filename = "co3d_leftout.pkl"
        pickle_file_path = os.path.join(dataset_root,samples_filename)

        self.all_samples = pd.read_pickle(pickle_file_path)
        self.classnames = []
        for classname in co3d_classnames:
            self.classnames += [template(classname) for template in co3d_template]  

    
    def __len__(self):
        return len(self.all_samples)

    def _load_16big_png_depth(self,depth_png):
        with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
        return depth

    def _load_depth(self,path, scale_adjustment):
        if not path.lower().endswith(".jpg.geometric.png"):
            raise ValueError('unsupported depth file name "%s"' % path)

        d = self._load_16big_png_depth(path) * scale_adjustment
        d[~np.isfinite(d)] = 0.0
        return d[None]  # fake feature channel

    def __getitem__(self, idx):
        
        sample = self.all_samples[idx]
        image_path = sample[0]
        depth_path = sample[1]
        scale = sample[2]
        mask_depth_path = sample[3]
        gt_pose = sample[4]
        sample_category = sample[5]

        image = self.transforms(Image.open(os.path.join(self.dataset_root,image_path)))
        #TODO: Crop depth
        depth = self._load_depth(os.path.join(self.dataset_root,depth_path),scale)

        r = Rot.from_matrix(gt_pose)
        angle = r.as_euler('yzx',degrees=True)[0]
        angle = abs(angle)

        granularity  = 18

        classes_per_category = len(co3d_template)

        quantized_angle = int(math.floor(angle/granularity))

        class_within_category = quantized_angle + 1

        category_index = co3d_classnames.index(sample_category)

        label1 = category_index*classes_per_category + class_within_category
        if self.mode == "train":
            label1_prob = 1.0
        else:
            label1_prob = 1.0

        label2 = label1 + 1
        if class_within_category == classes_per_category - 1 or self.mode == "val" or self.mode == "zeroshot":
            label2_prob = 0.0 
        else:
            label2_prob = 0.0

        label3 = label1 - 1
        if class_within_category == 1 or self.mode == "val" or self.mode == "zeroshot":
            label3_prob = 0.0 
        else:
            label3_prob = 0.0

        category_label = category_index*classes_per_category
        if self.mode == "train":
            category_label_prob = 0.0
        else:
            category_label_prob = 0.0

        train_val_target = torch.zeros(len(self.classnames),dtype=torch.float32)

        train_val_target[label1] = label1_prob
        if label2 < len(self.classnames):
            train_val_target[label2] = label2_prob
        train_val_target[label3] = label3_prob
        train_val_target[category_label] = category_label_prob



        target = train_val_target/torch.sum(train_val_target)


        return image, target

class Co3dDataset(Dataset):
    def __init__(self,transforms,dataset_root,mode,categories):
        logging.debug(f'Creating data from {dataset_root}.')
        self.dataset_root = dataset_root
        self.transforms = transforms
        if mode == "train":
            self.all_pairs = co3d_dataset_arrange.create_dataset_list(dataset_root,mode,categories,800000)
        else:
            self.all_pairs = co3d_dataset_arrange.create_dataset_list(dataset_root,mode,categories,80000)
    
    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):

        pair = self.all_pairs[idx]
        ref_image = self.transforms(Image.open(os.path.join(self.dataset_root,pair[2])))
        pred_image = self.transforms(Image.open(os.path.join(self.dataset_root,pair[4])))



        ref_rot = pair[5]
        pred_rot = pair[6]

        r1 = Rot.from_matrix(ref_rot)
        r2 = Rot.from_matrix(pred_rot)

        r1_inv = r1.inv()
        r_diff = r1_inv*r2
        r_diff = r_diff.as_euler('yzx',degrees=True)
        #r_diff = r_diff.as_rotvec()
        #norm_sq = np.dot(r_diff, r_diff)
        #angle = np.sqrt(norm_sq)
        angle = r_diff[0]

        if angle < 0:
            clockwise_angle = angle * (-1)
            counterclockwise_angle = 360 + angle
        else:
            clockwise_angle = 360 - angle
            counterclockwise_angle = angle
        angle = min(abs(clockwise_angle),abs(counterclockwise_angle))
        #angle = counterclockwise_angle

        #####angle for regression#####
        angle = angle/180
        #angle = angle/360
        angle = torch.Tensor(np.array([np.float32(angle)]))
        ##############################

        #####angle for classification#####
        #if(angle == 180):
        #    label = 8
        #else:
        #    label = int(np.floor(angle/20))
        
        #angle = torch.zeros(9,dtype=torch.float32)
        #angle[label] = 1
        ##################################
        

        path_ref = [pair[2]]
        path_rot = [pair[4]]

        text = tokenize(["a photo of a " + pair[7]])[0]

        return ref_image, pred_image, text, angle, path_ref, path_rot




        
    



class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]
        return images, texts


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # cc3m-train: 2905954
        # cc12m: 10968539
        # LAION-400m: 407332084
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader, sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption(sample):
    return 'txt' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_wds_dataset(args, preprocess_img, is_train, epoch=0):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    pipeline = [wds.SimpleShardList(input_shards)]
    # at this point we have an iterator over all the shards
    if is_train:
        pipeline.extend([
            wds.detshuffle(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=args.seed,
                epoch=epoch - 1,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
                rng=random.Random(args.seed),
            ),
            #wds.repeatedly,  # FIXME determine if this is beneficial
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        # roll over and repeat a few samples to get same number of full batches on each node
        global_batch_size = args.batch_size * args.world_size
        num_batches = math.ceil(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = math.ceil(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        # FIXME detshuffle uses same seed each epoch unless workers are persistent
        # this seems like a WDS bug, currently waiting for clarification
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_new_co3d_dataset(args, preprocess_fn, is_train, is_val, epoch=0):
    if is_train:
        input_filename = args.train_data
        mode = "train"
    else:
        if is_val:
            input_filename = args.val_data
            mode = "val"
        else:
            input_filename = args.zeroshot_data
            mode = "zeroshot"
    annot_filename = args.annot_data
    categories = args.categories
    assert input_filename
    dataset = Co3dDataset_New(
        preprocess_fn,
        input_filename,
        annot_filename,
        mode,
        categories)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
def get_co3d_dataset_ce(args, preprocess_fn, is_train, is_val_in, is_val_out, epoch=0):
    if is_train:
        input_filename = args.train_data
        mode = "train"
    elif is_val_in:
        input_filename = args.val_in_data
        mode = "val_in"
    elif is_val_out:
        input_filename = args.val_out_data
        mode = "val_out"
    else:
        input_filename = args.zeroshot_data
        mode = "zeroshot"
    annot_filename = args.annot_data
    categories = args.categories
    assert input_filename
    dataset = Co3dDataset_CE(
        preprocess_fn,
        input_filename,
        annot_filename,
        mode,
        categories)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_co3d_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    mode = "train" if is_train else "test"
    categories = args.categories
    assert input_filename
    dataset = Co3dDataset(
        preprocess_fn,
        input_filename,
        mode,
        categories)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "co3d":
        return get_co3d_dataset_ce
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, is_val_in=False, is_val_out=False)
    if args.val_in_data:
        data["val_in"] = get_dataset_fn(args.val_in_data, args.dataset_type)(
            args, preprocess_val, is_train=False, is_val_in=True,is_val_out=False)
    if args.val_in_data:
        data["val_out"] = get_dataset_fn(args.val_out_data, args.dataset_type)(
            args, preprocess_val, is_train=False, is_val_in=False, is_val_out=True)
    if args.zeroshot_data:
        data["zeroshot"] = get_dataset_fn(args.zeroshot_data, args.dataset_type)(
            args, preprocess_val, is_train=False, is_val_in=False,is_val_out=False)
    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
