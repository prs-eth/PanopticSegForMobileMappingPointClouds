import os
import os.path as osp
import numpy as np
from scipy import stats
from torch_points3d.models.panoptic.ply import read_ply, write_ply
import time
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
from tqdm.auto import tqdm as tq
import glob
import pandas as pd
from plyfile import PlyData, PlyElement

print('\nPreparing ply files')

raw_dir = 'data/s3disfused/raw'
cloud_names = 'Area_5'
folders = ["Area_{}".format(i) for i in range(1, 7)]
test_areas = [f for f in folders if cloud_names in f]
test_files = [
                (f, room_name, osp.join(raw_dir, f, room_name))
                for f in test_areas
                for room_name in os.listdir(osp.join(raw_dir, f))
                if os.path.isdir(osp.join(raw_dir, f, room_name))
            ]

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}
def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label

t0 = time.time()

#log directory
file_path_out = '/scratch2/torch-points3d/outputs/2021-10-13/15-22-11/eval/2021-10-26_00-16-21/'
#predicted semantic segmentation file path
pred_class_label_filenames = file_path_out+'Semantic_results_forEval.ply'
#predicted instance segmentation file path
pred_ins_label_filenames = file_path_out+'Instance_Embed_results_forEval.ply'
pred_ins_label_filenames_offset = file_path_out+'Instance_Offset_results_forEval.ply'
data_class = PlyData.read(pred_class_label_filenames)
data_ins = PlyData.read(pred_ins_label_filenames)
data_ins_offset = PlyData.read(pred_ins_label_filenames_offset)
pred_ins_complete = data_ins['vertex']['preds'].reshape(-1).astype(np.int)
pred_ins_complete_offset = data_ins_offset['vertex']['preds'].reshape(-1).astype(np.int)
pred_sem_complete = data_class['vertex']['preds'].reshape(-1).astype(np.int)
room_file_path = join(file_path_out +'prediction_perRoom_embed')
room_file_path2 = join(file_path_out +'prediction_perRoom_offset')
if not exists(room_file_path):
    makedirs(room_file_path)
if not exists(room_file_path2):
    makedirs(room_file_path2)
instance_count = 1
point_count = 0
print(test_files)
for (area, room_name, file_path) in tq(test_files):
#for cloud_name in cloud_names:
    
    #room_type = room_name.split("_")[0]
    # Initiate containers
    room_points = np.empty((0, 3), dtype=np.float32)
    room_colors = np.empty((0, 3), dtype=np.uint8)
    room_classes = np.empty((0, 1), dtype=np.int32)
    room_instances = np.empty((0, 1), dtype=np.int32)
    room_pre_ins = np.empty((0, 1), dtype=np.int32)
    room_pre_ins_offset = np.empty((0, 1), dtype=np.int32)
    room_pre_classes = np.empty((0, 1), dtype=np.int32)
    objects = glob.glob(osp.join(file_path, "Annotations/*.txt"))
    for single_object in objects:

        object_name = os.path.splitext(os.path.basename(single_object))[0]
        object_class = object_name.split("_")[0]
        object_label = object_name_to_label(object_class)
        object_data = pd.read_csv(single_object, sep=" ", header=None).values

        # Stack all data
        room_points = np.vstack((room_points, object_data[:, 0:3].astype(np.float32)))
        room_colors = np.vstack((room_colors, object_data[:, 3:6].astype(np.uint8)))
        object_classes = np.full((object_data.shape[0], 1), object_label, dtype=np.int32)
        room_classes = np.vstack((room_classes, object_classes))
        object_instances = np.full((object_data.shape[0], 1), instance_count, dtype=np.int32)
        room_instances = np.vstack((room_instances, object_instances))
        point_num_cur = np.shape(object_data)[0]
        pred_ins_cur = pred_ins_complete[point_count:point_count+point_num_cur].astype(np.uint8).reshape(-1,1)
        room_pre_ins = np.vstack((room_pre_ins, pred_ins_cur))
        pred_ins_cur_offset = pred_ins_complete_offset[point_count:point_count+point_num_cur].astype(np.uint8).reshape(-1,1)
        room_pre_ins_offset = np.vstack((room_pre_ins_offset, pred_ins_cur_offset))
        pred_sem_cur =  pred_sem_complete[point_count:point_count+point_num_cur].astype(np.uint8).reshape(-1,1)
        room_pre_classes = np.vstack((room_pre_classes, pred_sem_cur))
        point_count = point_count + point_num_cur
        instance_count = instance_count + 1
    
    room_file = file_path_out +'prediction_perRoom_embed/'+ cloud_names+ '_' + room_name + '.ply'
    print(room_file)
    # Save as ply
    write_ply(room_file,
                (room_points, room_colors, room_classes, room_instances, room_pre_classes, room_pre_ins),
                ['x', 'y', 'z', 'red', 'green', 'blue', 'gt_class', 'gt_ins', 'pre_sem', 'pre_ins'])
    room_file = file_path_out +'prediction_perRoom_offset/'+ cloud_names+ '_' + room_name + '.ply'
    print(room_file)
    # Save as ply
    write_ply(room_file,
                (room_points, room_colors, room_classes, room_instances, room_pre_classes, room_pre_ins_offset),
                ['x', 'y', 'z', 'red', 'green', 'blue', 'gt_class', 'gt_ins', 'pre_sem', 'pre_ins'])
    print(point_count)
    print(pred_ins_complete.shape)

print('Done in {:.1f}s'.format(time.time() - t0))