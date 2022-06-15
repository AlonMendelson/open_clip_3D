import pickle
import os
import json
import random
import gzip
import training.co3d_types as co3d_types
from typing import List

def create_dataset_list(dataset_root,mode,categories,known_prop = 1, unseen_prop = 1):
    all_pairs = []
    for category in categories:
        category_frameanot_path = os.path.join(dataset_root,category,"frame_annotations.jgz")
        with gzip.open(category_frameanot_path, "rt", encoding="utf8") as zipfile:
                frame_annots_list = co3d_types.load_dataclass(zipfile, List[co3d_types.FrameAnnotation])

        all_rotation_matrices = {}
        for frame in frame_annots_list:
            R = frame.viewpoint.R
            k = frame.sequence_name + "_" + str(frame.frame_number)
            all_rotation_matrices[k] = R
        json_root = os.path.join(dataset_root,category,"set_lists.json")
        json_file = open(json_root)
        data = json.load(json_file)
        known = data[mode+"_known"]
        unseen = data[mode+"_unseen"]
        all_frame_codes = []
        frame_start_index = []
        index = 0
        for frame1 in unseen:
            if not frame1[0] in all_frame_codes:
                all_frame_codes.append(frame1[0])
                frame_start_index.append(index)
            index += 1
        for frame1 in known:
            bit = random.uniform(0, 1)
            if bit > known_prop:
                continue
            frame_code = frame1[0]
            frame_index = all_frame_codes.index(frame_code)
            starting_index = frame_start_index[frame_index]
            if frame_index == len(frame_start_index) - 1:
                ending_index = len(unseen)
            else:
                ending_index = frame_start_index[frame_index + 1]
            R1 = all_rotation_matrices[frame_code + "_" + str(frame1[1])]
            for position in range(starting_index,ending_index):
                if unseen[position][0] == frame_code:
                    bit = random.uniform(0, 1)
                    if bit > unseen_prop:
                        continue
                    R2 = all_rotation_matrices[unseen[position][0] + "_" + str(unseen[position][1])]
                    new_pair = frame1 + unseen[position][1:] + [R1] + [R2] + [category]
                    all_pairs.append(new_pair)                

    random.shuffle(all_pairs)
    return all_pairs


def create_frame_annots_pickle(dataset_root, category):
    category_frameanot_path = os.path.join(dataset_root,category,"frame_annotations.jgz")
    with gzip.open(category_frameanot_path, "rt", encoding="utf8") as zipfile:
        frame_annots_list = co3d_types.load_dataclass(zipfile, List[co3d_types.FrameAnnotation])
    
    all_rotation_matrices = {}
    for frame in frame_annots_list:
        R = frame.viewpoint.R
        k = frame.sequence_name + "_" + str(frame.frame_number)
        all_rotation_matrices[k] = R

        
    zipfile.close()
    path = os.path.join(dataset_root,category,"rotations.pkl")
    with open(path, 'wb') as f:
        pickle.dump(all_rotation_matrices, f)



if __name__ == "__main__":
    create_dataset_list("../../../datasets/co3d","train",["teddybear"])
#   create_frame_annots_pickle("../../../datasets/co3d","teddybear")
