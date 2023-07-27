import os
import csv
import numpy as np
# from easydict import EasyDict as edict

# kpconv
# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold1_new/kpconv_fold1_new-KPConvPaperNPM3D-20220126_014633/eval/2022-01-31_10-33-25',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold2_new/kpconv_fold2_new-KPConvPaperNPM3D-20220126_060043/eval/2022-01-31_10-33-25',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold3/kpconv_fold3-KPConvPaperNPM3D-20220119_104625/eval/2022-01-27_01-43-00',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/kpconv_fold4/kpconv_fold4-KPConvPaperNPM3D-20220119_104625/eval/2022-01-28_00-38-31'
# ]

# minkov
# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold1/minkowski_fold1-MinkowskiBackboneNPM3D-20220118_161728/eval/2022-01-27_01-08-07',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold2/minkowski_fold2-MinkowskiBackboneNPM3D-20220118_161728/eval/2022-01-27_01-08-07',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold3/minkowski_fold3-MinkowskiBackboneNPM3D-20220118_163816/eval/2022-01-27_19-54-50',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold4/minkowski_fold4-MinkowskiBackboneNPM3D-20220118_173839/eval/2022-01-27_01-14-26'
# ]

# pointnet
# eval_paths = [
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold1/pointnet_fold1-pointnet2NPM3D-20220118_174257/eval/2022-01-27_15-20-46',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold2/pointnet_fold2-pointnet2NPM3D-20220119_005556/eval/2022-01-27_01-27-27',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold3/pointnet_fold3-pointnet2NPM3D-20220119_083227/eval/2022-01-27_01-27-27',
#     '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/pointnet_fold4/pointnet_fold4-pointnet2NPM3D-20220119_083515/eval/2022-01-28_00-05-23'
# ]

# test
eval_paths = [
    '/cluster/work/igp_psr/yuayue/RA/WP1/TP3D_PanopticSeg/outputs/minkowski_fold1/minkowski_fold1-MinkowskiBackboneNPM3D-20220118_161728/eval/2022-02-04_17-00-41'
]

things_idx = [2, 3, 4, 6, 7, 8]
stuff_idx = [0, 1, 5]
number_class = 9
number_things = 6
number_stuff = 3

result_offset_things_RQ = []
result_offset_things_SQ = []
result_offset_things_PQ = []
result_embeds_things_RQ = []
result_embeds_things_SQ = []
result_embeds_things_PQ = []
result_stuff_RQ = []
result_stuff_SQ = []
result_stuff_PQ = []
result_offset_RQ = []
result_offset_SQ = []
result_offset_PQ = []
result_offset_PQ_Star = []
result_embeds_RQ = []
result_embeds_SQ = []
result_embeds_PQ = []
result_embeds_PQ_Star = []


# read results
for eval_path in eval_paths:
    with open(eval_path + '/evaluation.txt') as f:
        for i, line in  enumerate(f):
            if ':' in line:
                (key, val) = line.split(':')
                if key=='Semantic Segmentation IoU':
                    semIoU = val.split('\n')[-2].split(", ")[1:9]
                    semIoU.append(val.split('\n')[-2].split(", ")[-1].split("]")[0])
                    semIoU = [float(i) for i in semIoU]
                    
                if key=='Instance Segmentation meanRQ':
                    num = float(val.split('\n')[-2])
                    if i <=24:
                        offset_things_meanRQ=num
                    else:
                        embeds_things_meanRQ=num
                if key=='Instance Segmentation meanSQ':
                    num = float(val.split('\n')[-2])
                    if i <=24:
                        offset_things_meanSQ=num
                    else:
                        embeds_things_meanSQ=num
                if key=='Instance Segmentation meanPQ':
                    num = float(val.split('\n')[-2])
                    if i <=24:
                        offset_things_meanPQ=num
                    else:
                        embeds_things_meanPQ=num

    stuff_RQ = []
    stuff_SQ = []
    stuff_PQ = []
    stuff_PQ_star = []

    for i in stuff_idx:
        if semIoU[i] >= 0.5:
            stuff_RQ.append(1)
            stuff_SQ.append(semIoU[i])
            stuff_PQ.append(semIoU[i])
        else:
            stuff_RQ.append(0)
            stuff_SQ.append(0)
            stuff_PQ.append(0)
        stuff_PQ_star.append(semIoU[i])
    
    stuff_meanRQ = np.mean(stuff_RQ)
    stuff_meanSQ = np.mean(stuff_SQ)
    stuff_meanPQ = np.mean(stuff_PQ)
    stuff_meanPQ_star = np.mean(stuff_PQ_star)

    offset_meanRQ = (offset_things_meanRQ*number_things+stuff_meanRQ*number_stuff)/number_class
    offset_meanSQ = (offset_things_meanSQ*number_things+stuff_meanSQ*number_stuff)/number_class
    offset_meanPQ = (offset_things_meanPQ*number_things+stuff_meanPQ*number_stuff)/number_class
    offset_meanPQ_star = (offset_things_meanPQ*number_things+stuff_meanPQ_star*number_stuff)/number_class

    embeds_meanRQ = (embeds_things_meanRQ*number_things+stuff_meanRQ*number_stuff)/number_class
    embeds_meanSQ = (embeds_things_meanSQ*number_things+stuff_meanSQ*number_stuff)/number_class
    embeds_meanPQ = (embeds_things_meanPQ*number_things+stuff_meanPQ*number_stuff)/number_class
    embeds_meanPQ_star = (embeds_things_meanPQ*number_things+stuff_meanPQ_star*number_stuff)/number_class
    
    result_offset_things_RQ.append(offset_things_meanRQ)
    result_offset_things_SQ.append(offset_things_meanSQ)
    result_offset_things_PQ.append(offset_things_meanPQ)
    result_embeds_things_RQ.append(embeds_things_meanRQ)
    result_embeds_things_SQ.append(embeds_things_meanSQ)
    result_embeds_things_PQ.append(embeds_things_meanPQ)
    result_stuff_RQ.append(stuff_meanRQ)
    result_stuff_SQ.append(stuff_meanSQ)
    result_stuff_PQ.append(stuff_meanPQ)
    result_offset_RQ.append(offset_meanRQ)
    result_offset_SQ.append(offset_meanSQ)
    result_offset_PQ.append(offset_meanPQ)
    result_offset_PQ_Star.append(offset_meanPQ_star)
    result_embeds_RQ.append(embeds_meanRQ)
    result_embeds_SQ.append(embeds_meanSQ)
    result_embeds_PQ.append(embeds_meanPQ)
    result_embeds_PQ_Star.append(embeds_meanPQ_star)

print('result_offset_things_RQ: {}'.format(np.mean(result_offset_things_RQ)))
print('result_offset_things_SQ: {}'.format(np.mean(result_offset_things_SQ)))
print('result_offset_things_PQ: {}'.format(np.mean(result_offset_things_PQ)))
print('result_embeds_things_RQ: {}'.format(np.mean(result_embeds_things_RQ)))                
print('result_embeds_things_SQ: {}'.format(np.mean(result_embeds_things_SQ)))
print('result_embeds_things_PQ: {}'.format(np.mean(result_embeds_things_PQ)))
print('result_stuff_RQ: {}'.format(np.mean(result_stuff_RQ)))                
print('result_stuff_SQ: {}'.format(np.mean(result_stuff_SQ)))
print('result_stuff_PQ: {}'.format(np.mean(result_stuff_PQ)))
print('result_offset_RQ: {}'.format(np.mean(result_offset_RQ)))                
print('result_offset_SQ: {}'.format(np.mean(result_offset_SQ)))
print('result_offset_PQ: {}'.format(np.mean(result_offset_PQ)))
print('result_offset_PQ_Star: {}'.format(np.mean(result_offset_PQ_Star)))
print('result_embeds_RQ: {}'.format(np.mean(result_embeds_RQ)))                
print('result_embeds_SQ: {}'.format(np.mean(result_embeds_SQ)))
print('result_embeds_PQ: {}'.format(np.mean(result_embeds_PQ)))
print('result_embeds_PQ_Star: {}'.format(np.mean(result_embeds_PQ_Star)))