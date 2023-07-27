import torch
import os
from torch_points_kernels import region_grow
from torch_geometric.data import Data
from torch_scatter import scatter
import random

from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import Seq, MLP, FastBatchNorm1d
from torch_points3d.core.losses import offset_loss, instance_iou_loss
from torch_points3d.core.data_transform import GridSampling3D
from .structures import PanopticLabels, PanopticResults
from torch_points3d.utils import is_list
from os.path import exists, join
from sklearn.cluster import MeanShift
from .ply import read_ply, write_ply
class PointGroup(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
    ]

    __REQUIRED_LABELS__ = list(PanopticLabels._fields)

    def __init__(self, option, model_type, dataset, modules):
        super(PointGroup, self).__init__(option)
        backbone_options = option.get("backbone", {"architecture": "unet"})
        self.Backbone = Minkowski(
            backbone_options.get("architecture", "unet"),
            input_nc=dataset.feature_dimension,
            num_layers=4,
            config=backbone_options.get("config", {}),
        )

        self._scorer_type = option.get("scorer_type", None)
        cluster_voxel_size = option.get("cluster_voxel_size", 0.05)
        if cluster_voxel_size:
            self._voxelizer = GridSampling3D(cluster_voxel_size, quantize_coords=True, mode="mean")
        else:
            self._voxelizer = None
        self.ScorerUnet = Minkowski("unet", input_nc=self.Backbone.output_nc, num_layers=4, config=option.scorer_unet)
        self.ScorerEncoder = Minkowski(
            "encoder", input_nc=self.Backbone.output_nc, num_layers=4, config=option.scorer_encoder
        )
        self.ScorerMLP = MLP([self.Backbone.output_nc, self.Backbone.output_nc, self.ScorerUnet.output_nc])
        self.ScorerHead = Seq().append(torch.nn.Linear(self.ScorerUnet.output_nc, 1)).append(torch.nn.Sigmoid())

        self.Offset = Seq().append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
        self.Offset.append(torch.nn.Linear(self.Backbone.output_nc, 3))

        self.Semantic = (
            Seq()
            .append(MLP([self.Backbone.output_nc, self.Backbone.output_nc], bias=False))
            .append(torch.nn.Linear(self.Backbone.output_nc, dataset.num_classes))
            .append(torch.nn.LogSoftmax(dim=-1))
        )
        self.loss_names = ["loss", "offset_norm_loss", "offset_dir_loss", "semantic_loss", "score_loss"]
        stuff_classes = dataset.stuff_classes
        if is_list(stuff_classes):
            stuff_classes = torch.Tensor(stuff_classes).long()
        self._stuff_classes = torch.cat([torch.tensor([IGNORE_LABEL]), stuff_classes])

    def get_opt_mergeTh(self):
        """returns configuration"""
        if self.opt.block_merge_th:
            return self.opt.block_merge_th
        else:
            return 0.01
    
    def set_input(self, data, device):
        self.raw_pos = data.pos.to(device)
        self.input = data
        all_labels = {l: data[l].to(device) for l in self.__REQUIRED_LABELS__}
        self.labels = PanopticLabels(**all_labels)

    def forward(self, epoch=-1, **kwargs):
        # Backbone
        backbone_features = self.Backbone(self.input).x

        # Semantic and offset heads
        semantic_logits = self.Semantic(backbone_features)
        offset_logits = self.Offset(backbone_features)

        # Grouping and scoring
        cluster_scores = None
        all_clusters = None
        cluster_type = None
        if epoch == -1 or epoch > self.opt.prepare_epoch:#>-1:  # Active by default
            all_clusters, cluster_type = self._cluster(semantic_logits, offset_logits)
            if len(all_clusters):
                cluster_scores = self._compute_score(all_clusters, backbone_features, semantic_logits)

        self.output = PanopticResults(
            semantic_logits=semantic_logits,
            offset_logits=offset_logits,
            clusters=all_clusters,
            cluster_scores=cluster_scores,
            cluster_type=cluster_type,
        )

        # Sets visual data for debugging
        #with torch.no_grad():
        #    self._dump_visuals(epoch)
        
        #with torch.no_grad():
        #    if epoch % 1 == 0:
        #        self._dump_visuals_fortest(epoch)

    def _cluster(self, semantic_logits, offset_logits):
        """ Compute clusters from positions and votes """
        predicted_labels = torch.max(semantic_logits, 1)[1]
        #clusters_pos = region_grow(
        #    self.raw_pos,
        #    predicted_labels,
        #    self.input.batch.to(self.device),
        #    ignore_labels=self._stuff_classes.to(self.device),
        #    radius=self.opt.cluster_radius_search,
        #    min_cluster_size=50
        #)
        clusters_pos =[]
        clusters_votes = region_grow(
            self.raw_pos + offset_logits,
            predicted_labels,
            self.input.batch.to(self.device),
            ignore_labels=self._stuff_classes.to(self.device),
            radius=self.opt.cluster_radius_search,
            nsample=200,
            min_cluster_size=50
        )

        all_clusters = clusters_pos + clusters_votes
        all_clusters = [c.to(self.device) for c in all_clusters]
        cluster_type = torch.zeros(len(all_clusters), dtype=torch.uint8).to(self.device)
        cluster_type[len(clusters_pos) :] = 1
        return all_clusters, cluster_type

    def _compute_score(self, all_clusters, backbone_features, semantic_logits):
        """ Score the clusters """
        if self._scorer_type:
            # Assemble batches
            x = []
            coords = []
            batch = []
            pos = []
            for i, cluster in enumerate(all_clusters):
                x.append(backbone_features[cluster])
                coords.append(self.input.coords[cluster])
                batch.append(i * torch.ones(cluster.shape[0]))
                pos.append(self.input.pos[cluster])
            batch_cluster = Data(x=torch.cat(x), coords=torch.cat(coords), batch=torch.cat(batch),)

            # Voxelise if required
            if self._voxelizer:
                batch_cluster.pos = torch.cat(pos)
                batch_cluster = batch_cluster.to(self.device)
                batch_cluster = self._voxelizer(batch_cluster)

            # Score
            batch_cluster = batch_cluster.to("cpu")
            if self._scorer_type == "MLP":
                score_backbone_out = self.ScorerMLP(batch_cluster.x.to(self.device))
                cluster_feats = scatter(
                    score_backbone_out, batch_cluster.batch.long().to(self.device), dim=0, reduce="max"
                )
            elif self._scorer_type == "encoder":
                score_backbone_out = self.ScorerEncoder(batch_cluster)
                cluster_feats = score_backbone_out.x
            else:
                score_backbone_out = self.ScorerUnet(batch_cluster)
                cluster_feats = scatter(
                    score_backbone_out.x, batch_cluster.batch.long().to(self.device), dim=0, reduce="max"
                )
            cluster_scores = self.ScorerHead(cluster_feats).squeeze(-1)
        else:
            # Use semantic certainty as cluster confidence
            with torch.no_grad():
                cluster_semantic = []
                batch = []
                for i, cluster in enumerate(all_clusters):
                    cluster_semantic.append(semantic_logits[cluster, :])
                    batch.append(i * torch.ones(cluster.shape[0]))
                cluster_semantic = torch.cat(cluster_semantic)
                batch = torch.cat(batch)
                cluster_semantic = scatter(cluster_semantic, batch.long().to(self.device), dim=0, reduce="mean")
                cluster_scores = torch.max(torch.exp(cluster_semantic), 1)[0]
        return cluster_scores

    def _compute_loss(self):
        # Semantic loss
        self.semantic_loss = torch.nn.functional.nll_loss(
            self.output.semantic_logits, (self.labels.y).to(torch.int64), ignore_index=IGNORE_LABEL
        )
        self.loss = self.opt.loss_weights.semantic * self.semantic_loss

        # Offset loss
        self.input.instance_mask = self.input.instance_mask.to(self.device)
        self.input.vote_label = self.input.vote_label.to(self.device)
        offset_losses = offset_loss(
            self.output.offset_logits[self.input.instance_mask],
            self.input.vote_label[self.input.instance_mask],
            torch.sum(self.input.instance_mask),
        )
        for loss_name, loss in offset_losses.items():
            setattr(self, loss_name, loss)
            self.loss += self.opt.loss_weights[loss_name] * loss

        # Score loss
        if self.output.cluster_scores is not None and self._scorer_type:
            self.score_loss = instance_iou_loss(
                self.output.clusters,
                self.output.cluster_scores,
                self.input.instance_labels.to(self.device),
                self.input.batch.to(self.device),
                min_iou_threshold=self.opt.min_iou_threshold,
                max_iou_threshold=self.opt.max_iou_threshold,
            )
            self.loss += self.score_loss * self.opt.loss_weights["score_loss"]

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self._compute_loss()
        self.loss.backward()

    def _dump_visuals(self, epoch):
        if random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            data_visual.vote = self.output.offset_logits
            nms_idx = self.output.get_instances()
            if self.output.clusters is not None:
                data_visual.clusters = [self.output.clusters[i].cpu() for i in nms_idx]
                data_visual.cluster_type = self.output.cluster_type[nms_idx]
            if not os.path.exists("viz"):
                os.mkdir("viz")
            torch.save(data_visual.to("cpu"), "viz/data_e%i_%i.pt" % (epoch, self.visual_count))
            self.visual_count += 1
    
    def _dump_visuals_fortest(self, epoch):
        if 0==self.opt.vizual_ratio: #random.random() < self.opt.vizual_ratio:
            if not hasattr(self, "visual_count"):
                self.visual_count = 0
            if not os.path.exists("viz"):
                os.mkdir("viz")
            if not os.path.exists("viz/epoch_%i" % (epoch)):
                os.mkdir("viz/epoch_%i" % (epoch))
            #if self.visual_count%10!=0:
            #    return
            print("epoch:{}".format(epoch))
            data_visual = Data(
                pos=self.raw_pos, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            )
            #    pos=self.input.coords, y=self.input.y, instance_labels=self.input.instance_labels, batch=self.input.batch
            data_visual.semantic_pred = torch.max(self.output.semantic_logits, -1)[1]
            #data_visual.embedding = self.output.embedding_logits
            data_visual.vote = self.output.offset_logits
            data_visual.semantic_prob = self.output.semantic_logits
            data_visual.input=self.input.x

            data_visual_fore = Data(
                pos=self.raw_pos[self.input.instance_mask], y=self.input.y[self.input.instance_mask], instance_labels=self.input.instance_labels[self.input.instance_mask], batch=self.input.batch[self.input.instance_mask],
                vote_label=self.labels.vote_label[self.input.instance_mask], 
                input=self.input.x[self.input.instance_mask]
            )
            data_visual_fore.vote = self.output.offset_logits[self.input.instance_mask]
            #data_visual_fore.embedding = self.output.embedding_logits[self.input.instance_mask]
            
            batch_size = torch.unique(data_visual_fore.batch)
            for s in batch_size:
                print(s)
                
                batch_mask_com = data_visual.batch == s
                example_name='example_complete_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                #write_ply(val_name,
                #            [data_visual.pos[batch_mask_com].detach().cpu().numpy(), 
                #            data_visual.y[batch_mask_com].detach().cpu().numpy().astype('int32'),
                #            data_visual.instance_labels[batch_mask_com].detach().cpu().numpy().astype('int32'),
                #            data_visual.semantic_prob[batch_mask_com].detach().cpu().numpy(),
                #            data_visual.embedding[batch_mask_com].detach().cpu().numpy(),
                #            data_visual.vote[batch_mask_com].detach().cpu().numpy().astype('int32'),
                #            data_visual.semantic_pred[batch_mask_com].detach().cpu().numpy().astype('int32'),
                #            data_visual.input[batch_mask_com].detach().cpu().numpy(),
                #            ],
                            #['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z','pre_ins_embed','pre_ins_offset', 'input_f1', 'input_f2', 'input_f3', 'input_f4', 'input_f5', 'input_f6', 'input_f7'])
                #            ['x', 'y', 'z', 'sem_label', 'ins_label',
                #            'sem_prob_1', 'sem_prob_2', 'sem_prob_3', 'sem_prob_4', 'sem_prob_5', 'sem_prob_6', 'sem_prob_7','sem_prob_8', 'sem_prob_9',
                            #'sem_prob_1', 'sem_prob_2', 'sem_prob_3', 'sem_prob_4',
                #            'embed_1', 'embed_2', 'embed_3', 'embed_4', 'embed_5',
                #            'offset_x_pre', 'offset_y_pre', 'offset_z_pre','sem_pre_1',
                #             'input_f1', 'input_f2', 'input_f3', 'input_f4','input_f5'])
                
                
                
                batch_mask = data_visual_fore.batch == s
                example_name='example_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
                #write_ply(val_name,
                #            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(), 
                #            data_visual_fore.y[batch_mask].detach().cpu().numpy().astype('int32'),
                #            data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                #            data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                #            data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                #            data_visual_fore.pre_ins[batch_mask].detach().cpu().numpy().astype('int32'),
                #            data_visual_fore.pre_ins2[batch_mask].detach().cpu().numpy().astype('int32'),
                #            data_visual_fore.input[batch_mask].detach().cpu().numpy(),
                #            ],
                            #['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z','pre_ins_embed','pre_ins_offset', 'input_f1', 'input_f2', 'input_f3', 'input_f4', 'input_f5', 'input_f6', 'input_f7'])
                #            ['x', 'y', 'z', 'sem_label', 'ins_label','offset_x', 'offset_y', 'offset_z', 'center_x', 'center_y', 'center_z','pre_ins_embed','pre_ins_offset', 'input_f1', 'input_f2', 'input_f3', 'input_f4', 'input_f5'])

                example_name='example_ins_{:d}'.format(self.visual_count)
                val_name = join("viz", "epoch_"+str(epoch), example_name)
            
                #clustering = MeanShift(bandwidth=self.opt.bandwidth).fit(data_visual_fore.embedding[batch_mask].detach().cpu())                        
                #pre_inslab = clustering.labels_
            
                write_ply(val_name,
                            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(), 
                            #data_visual_fore.embedding[batch_mask].detach().cpu().numpy(),
                            data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32'),
                            #pre_inslab.astype('int32'),
                            data_visual_fore.vote[batch_mask,0].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,0].detach().cpu().numpy(), 
                            data_visual_fore.vote[batch_mask,1].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,1].detach().cpu().numpy(), 
                            data_visual_fore.vote[batch_mask,2].detach().cpu().numpy(), 
                            data_visual_fore.vote_label[batch_mask,2].detach().cpu().numpy() 
                            ],
                            ['x', 'y', 'z', 'ins_label', 'offset_x', 'gt_offset_x', 'offset_y', 'gt_offset_y', 'offset_z', 'gt_offset_z'])
                #example_name = 'example_shiftedCorPre_{:d}'.format(self.visual_count)
                #val_name = join("viz", "epoch_"+str(epoch), example_name)

                #clustering = MeanShift(bandwidth=self.opt.bandwidth).fit((data_visual_fore.pos[batch_mask].detach().cpu()+data_visual_fore.vote[batch_mask].detach().cpu()))                    
                #pre_inslab = clustering.labels_
                #write_ply(val_name,
                #            [data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote[batch_mask].detach().cpu().numpy(),
                #             pre_inslab.astype('int32'),data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32')], 
                #            ['shifted_x_pre', 'shifted_y_pre', 'shifted_z_pre', 'pre_ins', 'ins_label'])
                
                #example_name = 'example_shiftedCorPreXYZ_{:d}'.format(self.visual_count)
                #val_name = join("viz", "epoch_"+str(epoch), example_name)
                #write_ply(val_name,
                #            [data_visual_fore.pos[batch_mask].detach().cpu().numpy(),
                #             pre_inslab.astype('int32'),data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32')], 
                #            ['x', 'y', 'z', 'pre_ins', 'ins_label'])
                
                #example_name = 'example_shiftedCorGT_{:d}'.format(self.visual_count)
                #val_name = join("viz", "epoch_"+str(epoch), example_name)
                #write_ply(val_name,
                #            [data_visual_fore.pos[batch_mask].detach().cpu().numpy()+data_visual_fore.vote_label[batch_mask].detach().cpu().numpy(),
                #             data_visual_fore.instance_labels[batch_mask].detach().cpu().numpy().astype('int32')],  
                #            ['shifted_x_gt', 'shifted_y_gt', 'shifted_z_gt', 'ins_label'])
                self.visual_count += 1
