from cmath import inf
import pickle
import os
import json
import random
import gzip
import training.co3d_types as co3d_types
import numpy as np
from typing import List
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot

front_views = {"backpack":6,"bicycle":4,"book":88,"car":71,"chair":5,"hairdryer":91,"hydrant":46,"keyboard":7,"laptop":18
                ,"motorcycle":30,"mouse":23,"remote":43,"teddybear":5,"toaster":16,"toilet":50,"toybus":5,"toyplane":71,"toytrain":17,"toytruck":3,"handbag":0}

validation_sequences = {"backpack":"372_41258_82188","bicycle":"350_36647_68462","book":"209_22171_45188","car":"106_12650_23736","hairdryer":"108_12886_25593","hydrant":"106_12698_26785","keyboard":"153_16970_32014","laptop":"112_13277_23636"
                ,"motorcycle":"216_22798_47409","mouse":"107_12753_23606","remote":"236_24795_52261","toaster":"274_29359_56269","toyplane":"264_28179_53215","toytrain":"372_41101_81827","toytruck":"346_36113_66551","handbag":"403_53234_103842"}
categories = ["hairdryer","car","teddybear","toilet",
    "toyplane","toytrain","toytruck","laptop","backpack","bicycle","chair",
    "keyboard","motorcycle","toybus","mouse","book"]
#categories = ["car","toilet","chair","toybus","teddybear"]
left_out_categories = ["chair","teddybear","toybus","toilet"]
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


def create_depth_dataset(dataset_root,sequence_filter_root,mode):
    all_samples = []
    left_out_samples = []
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
        if category == "book":
            reference_seq_code = "102_11955_20634"
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

        
        T_ai_cam = all_rotation_matrices[reference_seq_code + "_" + str(front_views[category])]
        T_oa = np.array(relevant_sequences[reference_seq_code]["trans"])
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                #seq to ref seq
                data = relevant_sequences[frame.sequence_name]
                T_ob = np.array(data["trans"])
                k = frame.sequence_name + "_" + str(frame.frame_number)
                T_bj_cam = all_rotation_matrices[k]
                GT_frame_pose = np.matmul(inv(T_oa),T_ai_cam)
                GT_frame_pose = np.matmul(T_ob,GT_frame_pose)
                GT_frame_pose = np.matmul(inv(T_bj_cam),GT_frame_pose)
                GT_rot = GT_frame_pose[0:3,0:3]
                r = Rot.from_matrix(GT_rot)
                angle2 = abs(r.as_euler("yzx",degrees=True)[0])
                if category in left_out_categories and angle2>=18:
                    left_out_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
                else:
                    all_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
    
    random.shuffle(all_samples)
    total_num_samples = len(all_samples)
    samples_for_train = int(0.9*total_num_samples)
    samples_for_val = total_num_samples-samples_for_train
    path = os.path.join(dataset_root,"co3d_train_red_val_in.pkl")
    with open(path, 'wb') as f:
        pickle.dump(all_samples[0:samples_for_train], f)
    path = os.path.join(dataset_root,"co3d_val_red_val_in.pkl")
    with open(path, 'wb') as f:
        pickle.dump(all_samples[-samples_for_val:], f)
    path = os.path.join(dataset_root,"co3d_leftout_red_val_in.pkl")
    with open(path, 'wb') as f:
        pickle.dump(left_out_samples, f)


def create_depth_dataset_new_val(dataset_root,sequence_filter_root,mode):
    train_samples = []
    val_samples = []
    left_out_samples = []
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
        if category == "book":
            reference_seq_code = "102_11955_20634"
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

        
        T_ai_cam = all_rotation_matrices[reference_seq_code + "_" + str(front_views[category])]
        T_oa = np.array(relevant_sequences[reference_seq_code]["trans"])
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                #seq to ref seq
                data = relevant_sequences[frame.sequence_name]
                T_ob = np.array(data["trans"])
                k = frame.sequence_name + "_" + str(frame.frame_number)
                T_bj_cam = all_rotation_matrices[k]
                GT_frame_pose = np.matmul(inv(T_oa),T_ai_cam)
                GT_frame_pose = np.matmul(T_ob,GT_frame_pose)
                GT_frame_pose = np.matmul(inv(T_bj_cam),GT_frame_pose)
                GT_rot = GT_frame_pose[0:3,0:3]
                r = Rot.from_matrix(GT_rot)
                angle2 = abs(r.as_euler("yzx",degrees=True)[0])
                if category in left_out_categories and angle2 >= 18:
                    left_out_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
                else:
                    if category in validation_sequences and frame.sequence_name == validation_sequences[category]:
                        val_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
                    else:
                        train_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
    
    random.shuffle(val_samples)
    random.shuffle(train_samples)
    path = os.path.join(dataset_root,"co3d_train_red_val_out_car.pkl")
    with open(path, 'wb') as f:
        pickle.dump(train_samples, f)
    path = os.path.join(dataset_root,"co3d_val_red_val_out_car.pkl")
    with open(path, 'wb') as f:
        pickle.dump(val_samples, f)
    path = os.path.join(dataset_root,"co3d_leftout_red_val_out_car.pkl")
    with open(path, 'wb') as f:
        pickle.dump(left_out_samples, f)


def create_dataset_4_splits(dataset_root,sequence_filter_root,mode):
    random.seed(42)
    train_samples = []
    in_val_samples = []
    out_val_samples = []
    left_out_samples = []
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
        if category == "book":
            reference_seq_code = "102_11955_20634"
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

        
        T_ai_cam = all_rotation_matrices[reference_seq_code + "_" + str(front_views[category])]
        T_oa = np.array(relevant_sequences[reference_seq_code]["trans"])
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                #seq to ref seq
                data = relevant_sequences[frame.sequence_name]
                T_ob = np.array(data["trans"])
                k = frame.sequence_name + "_" + str(frame.frame_number)
                T_bj_cam = all_rotation_matrices[k]
                GT_frame_pose = np.matmul(inv(T_oa),T_ai_cam)
                GT_frame_pose = np.matmul(T_ob,GT_frame_pose)
                GT_frame_pose = np.matmul(inv(T_bj_cam),GT_frame_pose)
                GT_rot = GT_frame_pose[0:3,0:3]
                r = Rot.from_matrix(GT_rot)
                angle2 = abs(r.as_euler("yzx",degrees=True)[0])
                if category in left_out_categories and angle2 >= 18:
                    left_out_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
                else:
                    if category in validation_sequences and frame.sequence_name == validation_sequences[category]:
                        out_val_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
                    else:
                        p = random.uniform(0, 1)
                        if p <= 0.1:
                            in_val_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
                        else:
                            train_samples.append([frame.image.path,frame.depth.path,frame.depth.scale_adjustment,frame.depth.mask_path,GT_rot,category])
    
    random.shuffle(out_val_samples)
    random.shuffle(in_val_samples)
    random.shuffle(train_samples)
    path = os.path.join(dataset_root,"co3d_train.pkl")
    with open(path, 'wb') as f:
        pickle.dump(train_samples, f)
    path = os.path.join(dataset_root,"co3d_val_out.pkl")
    with open(path, 'wb') as f:
        pickle.dump(out_val_samples, f)
    path = os.path.join(dataset_root,"co3d_val_in.pkl")
    with open(path, 'wb') as f:
        pickle.dump(in_val_samples, f)
    path = os.path.join(dataset_root,"co3d_leftout.pkl")
    with open(path, 'wb') as f:
        pickle.dump(left_out_samples, f)


def create_dataset_4_splits_left_right(dataset_root,sequence_filter_root,mode):
    random.seed(42)
    train_samples = []
    in_val_samples = []
    out_val_samples = []
    left_out_samples = []
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
        if category == "book":
            reference_seq_code = "102_11955_20634"
        all_rotation_matrices = {}
        index = 0
        last_sequence = "None"
        sequence_starts = {}
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                if frame.sequence_name != last_sequence:
                    sequence_starts[frame.sequence_name] = index
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
            index += 1
            last_sequence = frame.sequence_name

        
        T_ai_cam = all_rotation_matrices[reference_seq_code + "_" + str(front_views[category])]
        T_oa = np.array(relevant_sequences[reference_seq_code]["trans"])
        for frame in frame_annots_list:
            if frame.sequence_name in relevant_sequences:
                #seq to ref seq
                data = relevant_sequences[frame.sequence_name]
                T_ob = np.array(data["trans"])
                k = frame.sequence_name + "_" + str(frame.frame_number)
                T_bj_cam = all_rotation_matrices[k]
                GT_frame_pose = np.matmul(inv(T_oa),T_ai_cam)
                GT_frame_pose = np.matmul(T_ob,GT_frame_pose)
                GT_frame_pose = np.matmul(inv(T_bj_cam),GT_frame_pose)
                GT_rot = GT_frame_pose[0:3,0:3]
                r = Rot.from_matrix(GT_rot)
                angle2 = abs(r.as_euler("yzx",degrees=True)[0])
                angle = r.as_euler("yzx",degrees=True)[0] if r.as_euler("yzx",degrees=True)[0] >= 0 else 360 + r.as_euler("yzx",degrees=True)[0]
                start_ind = sequence_starts[frame.sequence_name]
                before_path = ""
                after_path = ""
                before_dist = inf
                after_dist = inf
                check_frame = frame_annots_list[start_ind]
                curr_sequence_name = check_frame.sequence_name
                while curr_sequence_name == frame.sequence_name:
                    if check_frame.frame_number == frame.frame_number:
                        start_ind += 1
                        if start_ind == len(frame_annots_list):
                            break
                        check_frame = frame_annots_list[start_ind]
                        curr_sequence_name = check_frame.sequence_name
                        continue
                    k_tag = check_frame.sequence_name + "_" + str(check_frame.frame_number)
                    T_bj_cam_tag = all_rotation_matrices[k_tag]
                    GT_frame_pose_tag = np.matmul(inv(T_oa),T_ai_cam)
                    GT_frame_pose_tag = np.matmul(T_ob,GT_frame_pose_tag)
                    GT_frame_pose_tag = np.matmul(inv(T_bj_cam_tag),GT_frame_pose_tag)
                    GT_rot_tag = GT_frame_pose_tag[0:3,0:3]
                    r_tag = Rot.from_matrix(GT_rot_tag)
                    angle_tag = r_tag.as_euler("yzx",degrees=True)[0] if r_tag.as_euler("yzx",degrees=True)[0] >= 0 else 360 + r_tag.as_euler("yzx",degrees=True)[0]
                    angle_copy = angle
                    if angle_copy <= 10 and angle_tag >= 350:
                        angle_tag = angle_tag - 360
                    if angle_copy >= 350 and angle_tag <= 10:
                        angle_copy = angle_copy - 360
                    if angle_tag < angle_copy:
                        dist = angle_copy - angle_tag
                        if dist < before_dist:
                            before_dist = dist
                            before_path = check_frame.image.path
                    if angle_tag > angle_copy:
                        dist = angle_tag - angle_copy
                        if dist < after_dist:
                            after_dist = dist
                            after_path = check_frame.image.path
                    start_ind += 1
                    if start_ind == len(frame_annots_list):
                        break
                    check_frame = frame_annots_list[start_ind]
                    curr_sequence_name = check_frame.sequence_name
                if before_path != "" and after_path != "":
                    train_valid = True
                else:
                    train_valid = False
                    
                if category in left_out_categories and angle2 >= 18:
                    left_out_samples.append([frame.image.path,GT_rot,category])
                else:
                    if category in validation_sequences and frame.sequence_name == validation_sequences[category]:
                        out_val_samples.append([frame.image.path,GT_rot,category])
                    else:
                        p = random.uniform(0, 1)
                        if p <= 0.1 or train_valid == False:
                            in_val_samples.append([frame.image.path,GT_rot,category])
                        else:
                            train_samples.append([frame.image.path,before_path,before_dist,after_path,after_dist,GT_rot,category])
    
    random.shuffle(out_val_samples)
    random.shuffle(in_val_samples)
    random.shuffle(train_samples)
    path = os.path.join(dataset_root,"co3d_train_lr.pkl")
    with open(path, 'wb') as f:
        pickle.dump(train_samples, f)
    path = os.path.join(dataset_root,"co3d_val_out_lr.pkl")
    with open(path, 'wb') as f:
        pickle.dump(out_val_samples, f)
    path = os.path.join(dataset_root,"co3d_val_in_lr.pkl")
    with open(path, 'wb') as f:
        pickle.dump(in_val_samples, f)
    path = os.path.join(dataset_root,"co3d_leftout_lr.pkl")
    with open(path, 'wb') as f:
        pickle.dump(left_out_samples, f)


if __name__ == "__main__":
#    create_dataset_list("../../../datasets/co3d","train",["stopsign"],500000)
#   create_frame_annots_pickle("../../../datasets/co3d","teddybear")
#    create_depth_dataset_new_val("../../../datasets/co3d","../zero-shot-pose/data/class_labels","train")
    create_dataset_4_splits_left_right("../../../datasets/co3d","../zero-shot-pose/data/class_labels","train")
