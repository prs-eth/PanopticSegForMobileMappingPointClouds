import os
import numpy as np
from scipy import stats
from torch_points3d.models.panoptic.ply import read_ply, write_ply
from plyfile import PlyData, PlyElement
NUM_CLASSES = 10
NUM_CLASSES_count = 9
#class index for instance segmenatation
ins_classcount = [3,4,5,7,8,9] 
#class index for semantic segmenatation
sem_classcount = [1,2,3,4,5,6,7,8,9] 

#log directory
file_path = '/scratch2/torch-points3d/outputs/2021-10-20/06-19-43/eval/2021-10-26_14-27-55/'
#predicted semantic segmentation file path
pred_class_label_filename = file_path+'Semantic_results_forEval.ply'
#predicted instance segmentation file path
pred_ins_label_filename = file_path+'Instance_Offset_results_forEval.ply'

# Initialize...
LOG_FOUT = open(os.path.join(file_path+'evaluation.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
# acc and macc
true_positive_classes = np.zeros(NUM_CLASSES)
positive_classes = np.zeros(NUM_CLASSES)
gt_classes = np.zeros(NUM_CLASSES)

# precision & recall
total_gt_ins = np.zeros(NUM_CLASSES)
at = 0.5
tpsins = [[] for itmp in range(NUM_CLASSES)]
fpsins = [[] for itmp in range(NUM_CLASSES)]
# mucov and mwcov
all_mean_cov = [[] for itmp in range(NUM_CLASSES)]
all_mean_weighted_cov = [[] for itmp in range(NUM_CLASSES)]


#read files
data_class = PlyData.read(pred_class_label_filename)
data_ins = PlyData.read(pred_ins_label_filename)

pred_ins_complete = data_ins['vertex']['preds'].reshape(-1).astype(np.int)
pred_sem_complete = data_class['vertex']['preds'].reshape(-1).astype(np.int)+1
gt_ins_complete = data_ins['vertex']['gt'].reshape(-1).astype(np.int)
gt_sem_complete = data_class['vertex']['gt'].reshape(-1).astype(np.int)+1

idxc = ((gt_sem_complete!=0) & (gt_sem_complete!=1) & (gt_sem_complete!=2) &  (gt_sem_complete!=6)) | ((pred_sem_complete!=0) & (pred_sem_complete!=1) & (pred_sem_complete!=2) &  (pred_sem_complete!=6))
pred_ins = pred_ins_complete[idxc]
gt_ins = gt_ins_complete[idxc]
pred_sem = pred_sem_complete[idxc]
gt_sem = gt_sem_complete[idxc]

# pn semantic mIoU
for j in range(gt_sem_complete.shape[0]):
    gt_l = int(gt_sem_complete[j])
    pred_l = int(pred_sem_complete[j])
    gt_classes[gt_l] += 1
    positive_classes[pred_l] += 1
    true_positive_classes[gt_l] += int(gt_l==pred_l) 

# semantic results
iou_list = []
for i in range(NUM_CLASSES):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    iou_list.append(iou)

log_string('Semantic Segmentation oAcc: {}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
#log_string('Semantic Segmentation Acc: {}'.format(true_positive_classes / gt_classes))
log_string('Semantic Segmentation mAcc: {}'.format(np.mean(true_positive_classes[sem_classcount] / gt_classes[sem_classcount])))
log_string('Semantic Segmentation IoU: {}'.format(iou_list))
log_string('Semantic Segmentation mIoU: {}'.format(1.*sum(iou_list)/NUM_CLASSES_count))

# instance
un = np.unique(pred_ins)
pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
for ig, g in enumerate(un):  # each object in prediction
    if g == -1:
        continue
    tmp = (pred_ins == g)
    sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
    pts_in_pred[sem_seg_i] += [tmp]

un = np.unique(gt_ins)
pts_in_gt = [[] for itmp in range(NUM_CLASSES)]
for ig, g in enumerate(un):
    if g == -1:
        continue
    tmp = (gt_ins == g)
    sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
    pts_in_gt[sem_seg_i] += [tmp]

# instance mucov & mwcov
for i_sem in range(NUM_CLASSES):
    sum_cov = 0
    mean_cov = 0
    mean_weighted_cov = 0
    num_gt_point = 0
    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
        ovmax = 0.
        num_ins_gt_point = np.sum(ins_gt)
        num_gt_point += num_ins_gt_point
        for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
            union = (ins_pred | ins_gt)
            intersect = (ins_pred & ins_gt)
            iou = float(np.sum(intersect)) / np.sum(union)

            if iou > ovmax:
                ovmax = iou
                ipmax = ip

        sum_cov += ovmax
        mean_weighted_cov += ovmax * num_ins_gt_point

    if len(pts_in_gt[i_sem]) != 0:
        mean_cov = sum_cov / len(pts_in_gt[i_sem])
        all_mean_cov[i_sem].append(mean_cov)

        mean_weighted_cov /= num_gt_point
        all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

#print(all_mean_cov)

# instance precision & recall
for i_sem in range(NUM_CLASSES):
    tp = [0.] * len(pts_in_pred[i_sem])
    fp = [0.] * len(pts_in_pred[i_sem])
    gtflag = np.zeros(len(pts_in_gt[i_sem]))
    total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

    for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
        ovmax = -1.

        for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
            union = (ins_pred | ins_gt)
            intersect = (ins_pred & ins_gt)
            iou = float(np.sum(intersect)) / np.sum(union)

            if iou > ovmax:
                ovmax = iou
                igmax = ig

        if ovmax >= at:
                tp[ip] = 1  # true
        else:
            fp[ip] = 1  # false positive

    tpsins[i_sem] += tp
    fpsins[i_sem] += fp


MUCov = np.zeros(NUM_CLASSES)
MWCov = np.zeros(NUM_CLASSES)
for i_sem in range(NUM_CLASSES):
    MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
    MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

precision = np.zeros(NUM_CLASSES)
recall = np.zeros(NUM_CLASSES)
for i_sem in range(NUM_CLASSES):
    tp = np.asarray(tpsins[i_sem]).astype(np.float)
    fp = np.asarray(fpsins[i_sem]).astype(np.float)
    tp = np.sum(tp)
    fp = np.sum(fp)
    rec = tp / total_gt_ins[i_sem]
    prec = tp / (tp + fp)

    precision[i_sem] = prec
    recall[i_sem] = rec

# instance results
log_string('Instance Segmentation MUCov: {}'.format(MUCov[ins_classcount]))
log_string('Instance Segmentation mMUCov: {}'.format(np.mean(MUCov[ins_classcount])))
log_string('Instance Segmentation MWCov: {}'.format(MWCov[ins_classcount]))
log_string('Instance Segmentation mMWCov: {}'.format(np.mean(MWCov[ins_classcount])))
log_string('Instance Segmentation Precision: {}'.format(precision[ins_classcount]))
log_string('Instance Segmentation mPrecision: {}'.format(np.mean(precision[ins_classcount])))
log_string('Instance Segmentation Recall: {}'.format(recall[ins_classcount]))
log_string('Instance Segmentation mRecall: {}'.format(np.mean(recall[ins_classcount])))
