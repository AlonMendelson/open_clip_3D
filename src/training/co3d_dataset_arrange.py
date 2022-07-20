import pickle
import os
import json
import random
import gzip
import co3d_types as co3d_types
import numpy as np
from typing import List
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

def create_dataset_list_split(dataset_root,mode,categories,  known_prop = 1, unseen_prop = 1):
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

def create_dataset_list(dataset_root,mode,categories, num_samples):
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
        all_instances = []
        json_root = os.path.join(dataset_root,category,"set_lists.json")
        json_file = open(json_root)
        data = json.load(json_file)
        known = data[mode+"_known"]
        unseen = data[mode+"_unseen"]
        i = 0
        j = 0
        while i  < (len(known)):
            sequence_code = known[i][0]
            instance_frames = []
            while known[i][0] == sequence_code:
                instance_frames.append(known[i])
                i += 1
                if i == len(known):
                    break
            while unseen[j][0] == sequence_code:
                instance_frames.append(unseen[j])
                j += 1
                if j == len(unseen):
                    break
            all_instances.append(instance_frames)
        for instance in all_instances:
            for i in range(len(instance)):
                for j in range(i+1,len(instance)):
                    pair = instance[i] + instance[j][1:] + [all_rotation_matrices[instance[i][0] + "_" + str(instance[i][1])]] + [all_rotation_matrices[instance[j][0] + "_" + str(instance[j][1])]] + [category]
                    all_pairs.append(pair)
    random.shuffle(all_pairs)
    if num_samples > len(all_pairs):
        num_samples = len(all_pairs)
    return all_pairs[0:num_samples]


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


def create_depth_dataset(dataset_root,sequence_filter_root,categories):
    all_samples = []
    for category in categories:
        category_frameanot_path = os.path.join(dataset_root,category,"frame_annotations.jgz")
        with gzip.open(category_frameanot_path, "rt", encoding="utf8") as zipfile:
                frame_annots_list = co3d_types.load_dataclass(zipfile, List[co3d_types.FrameAnnotation])
        
        relevant_sequences = {}
        reference_seq_code = 0
        annotations_dir = os.path.join(sequence_filter_root,category)
        for sequence in os.listdir(annotations_dir):
            #get sequence code
            sequence_code = sequence.split(".")[0]
            #get sequence json file
            sequence_json = os.path.join(annotations_dir,sequence)
            #open json file
            json_file = open(sequence_json)
            #get data
            data = json.load(json_file)
            relevant_sequences[sequence_code] = data
            reference_seq_code = data["reference_seq"]

        all_rotation_matrices = {}
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                R = np.array(frame.viewpoint.R)
                T = np.array(frame.viewpoint.T)
                M = np.zeros((4,4))
                M[0][0] = R[0][0]
                M[0][1] = R[0][1]
                M[0][2] = R[0][2]
                M[1][0] = R[1][0]
                M[1][1] = R[1][1]
                M[1][2] = R[1][2]
                M[2][0] = R[2][0]
                M[2][1] = R[2][1]
                M[2][2] = R[2][2]
                M[0][3] = T[0]
                M[1][3] = T[1]
                M[2][3] = T[2]
                M[3][3] = 1
                k = frame.sequence_name + "_" + str(frame.frame_number)
                all_rotation_matrices[k] = M
        
        reference_rotation = inv(all_rotation_matrices[reference_seq_code + "_" + str(0)])
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                #seq to ref seq
                data = relevant_sequences[frame.sequence_name]
                seq2seq = np.array(data["trans"])
                k = frame.sequence_name + "_" + str(frame.frame_number)
                frame_pose = all_rotation_matrices[k]
                #a_i = ref, b_j = frame
                GT_frame_pose = np.matmul(np.matmul(frame_pose,seq2seq),reference_rotation)
                GT_rot = GT_frame_pose[0:3,0:3]
                r = Rot.from_matrix(GT_rot)
                a = r.as_euler('yzx', degrees=True)
                all_samples.append([frame,GT_frame_pose])
    
    return all_samples


if __name__ == "__main__":
#    create_dataset_list("../../../datasets/co3d","train",["stopsign"],500000)
#   create_frame_annots_pickle("../../../datasets/co3d","teddybear")
    create_depth_dataset("../../../datasets/co3d","../zero-shot-pose/data/class_labels",["toyplane"])
