import os
import csv
import numpy as np
from easydict import EasyDict as edict

# input path
# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold1/minkowski_fold1-MinkowskiBackboneNPM3D-20220118_161728/eval/2022-01-27_01-08-07',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold2/minkowski_fold2-MinkowskiBackboneNPM3D-20220118_161728/eval/2022-01-27_01-08-07',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold3/minkowski_fold3-MinkowskiBackboneNPM3D-20220118_163816/eval/2022-01-27_19-54-50',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold4/minkowski_fold4-MinkowskiBackboneNPM3D-20220118_173839/eval/2022-01-27_01-14-26'
# ]

# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold1/pointnet_fold1-pointnet2NPM3D-20220118_174257/eval/2022-01-27_15-20-46',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold2/pointnet_fold2-pointnet2NPM3D-20220119_005556/eval/2022-01-27_01-27-27',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold3/pointnet_fold3-pointnet2NPM3D-20220119_083227/eval/2022-01-27_01-27-27',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold4/pointnet_fold4-pointnet2NPM3D-20220119_083515/eval/2022-01-28_00-05-23'
# ]

# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold1_new/kpconv_fold1_new-KPConvPaperNPM3D-20220126_014633/eval/2022-01-31_10-33-25',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold2_new/kpconv_fold2_new-KPConvPaperNPM3D-20220126_060043/eval/2022-01-31_10-33-25',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold3/kpconv_fold3-KPConvPaperNPM3D-20220119_104625/eval/2022-01-27_01-43-00',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold4/kpconv_fold4-KPConvPaperNPM3D-20220119_104625/eval/2022-01-28_00-38-31'
# ]

# kpconv
eval_paths = [
     '/cluster/work/igp_psr/binbin/torch-points3d/outputs/pointgroup-003-area1-right/pointgroup-003-area1-right-PointGroup-PAPER-20220323_171137/eval/2022-03-30_10-09-16',
     '/cluster/work/igp_psr/binbin/torch-points3d/outputs/pointgroup-003-area2-right/pointgroup-003-area2-right-PointGroup-PAPER-20220323_171137/eval/2022-03-30_11-41-03',
     '/cluster/work/igp_psr/binbin/torch-points3d/outputs/pointgroup-003-area3-right/pointgroup-003-area3-right-PointGroup-PAPER-20220323_171137/eval/2022-03-30_13-03-49',
     '/cluster/work/igp_psr/binbin/torch-points3d/outputs/pointgroup-003-area4-right/pointgroup-003-area4-right-PointGroup-PAPER-20220323_171137/eval/2022-03-30_13-50-37',
     '/cluster/work/igp_psr/binbin/torch-points3d/outputs/pointgroup-003-area5/pointgroup-003-area5-PointGroup-PAPER-20220321_134704/eval/2022-03-30_15-02-44',
     '/cluster/work/igp_psr/binbin/torch-points3d/outputs/pointgroup-003-area6/pointgroup-003-area6-PointGroup-PAPER-20220321_143334/eval/2022-03-30_17-08-17'
     ]

# minkow
# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold1_v2_500/pointnet_fold1_v2_500-pointnet2NPM3D-20220208_113018/eval/2022-02-11_14-56-22',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold2_v2_500/pointnet_fold2_v2_500-pointnet2NPM3D-20220208_115341/eval/2022-02-11_14-56-22',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold3_v2_500/pointnet_fold3_v2_500-pointnet2NPM3D-20220208_115942/eval/2022-02-11_14-56-22',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/'
# ]

# pointnet
#eval_paths = [
#    '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold1_v2_500/pointnet_fold1_v2_500-pointnet2NPM3D-20220208_113018/eval/2022-02-11_14-56-22',
#    '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold2_v2_500/pointnet_fold2_v2_500-pointnet2NPM3D-20220208_115341/eval/2022-02-11_14-56-22',
#    '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold3_v2_500/pointnet_fold3_v2_500-pointnet2NPM3D-20220208_115942/eval/2022-02-11_14-56-22',
#    '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold4_v2_500/pointnet_fold4_v2_500-pointnet2NPM3D-20220208_122456/eval/2022-02-12_20-13-52'
#]
# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold4_v2_500/pointnet_fold4_v2_500-pointnet2NPM3D-20220208_122456/eval/2022-02-12_20-13-52'
# ]
# output result path
csv_result = 'Kpconv_S3DIS.csv'

result = edict({
    'Semantic Segmentation oAcc':[], 
    'Semantic Segmentation mAcc':[],
    'Semantic Segmentation mIoU': [],
    'Instance Segmentation mMUCov':{'offset':[], 'embedding':[]},
    'Instance Segmentation mMWCov':{'offset':[], 'embedding':[]},
    'Instance Segmentation mPrecision':{'offset':[], 'embedding':[]},
    'Instance Segmentation mRecall':{'offset':[], 'embedding':[]},
    'Instance Segmentation F1 score':{'offset':[], 'embedding':[]},
    'Instance Segmentation meanRQ':{'offset':[], 'embedding':[]},
    'Instance Segmentation meanSQ':{'offset':[], 'embedding':[]},
    'Instance Segmentation meanPQ':{'offset':[], 'embedding':[]}
    })

# read results
for eval_path in eval_paths:
    with open(eval_path + '/evaluation.txt') as f:
        for i, line in  enumerate(f):
            if ':' in line:
                (key, val) = line.split(':')
                for target_key in result.keys():
                    if target_key == key:
                        num = float(val.split('\n')[-2])
                        if (i>=45) and (i<=50):
                            result[target_key].append(num)
                        elif i<=22:
                            result[target_key].offset.append(num)
                        elif (i>=23) and (i<=44):
                            result[target_key].embedding.append(num)

# average results
for key in result.keys():
    if key.startswith('Semantic'):
        result[key] = round(sum(result[key]) / len(result[key]),4)
    elif key.startswith('Instance'):
        result[key].offset = round(sum(result[key].offset) / len(result[key].offset),4)
        result[key].embedding = round(sum(result[key].embedding) / len(result[key].embedding),4)

# write results
with open(csv_result, 'w') as file:
    writer = csv.writer(file)
    writer.writerows(result.items())
     
            
                

                    



