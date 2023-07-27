# A Review of Panoptic Segmentation for Mobile Mapping Point Clouds

This repository represents the official code for paper entitled "A Review of Panoptic Segmentation for Mobile Mapping Point Clouds". 

<p align="center">
  <img width="70%" src="/figures/PanopticSegExample.png" />
</p>

# Set up environment

The framework used in this code is torchpoints-3d, so generally the installation instructions for torchpoints-3d can follow the official ones: 

https://torch-points3d.readthedocs.io/en/latest/

https://github.com/torch-points3d/torch-points3d

Here are two detailed examples for installation worked on our local computers for your reference:

### Example 1 of installation

Specs local computer: Ubuntu 22.04, 64-bit, CUDA version 11.7 -> but CUDA is backwards compatible, so here we used CUDA 11.1 for all libraries installed.

Commands in terminal using miniconda:
```bash
conda create -n treeins_env_local python=3.8

conda activate treeins_env_local

conda install pytorch=1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia

pip install numpy==1.19.5

conda install openblas-devel -c anaconda

export CUDA_HOME=/usr/local/cuda-11

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" –install-option="--blas=openblas"

#CHECK IF TORCH AND MINKOWSKI ENGINE WORK AS EXPECTED:
(treeins_env_local) : python
Python 3.8.13 (default, #DATE#, #TIME#)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> import MinkowskiEngine
>>> exit()
#CHECK FINISHED

pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install torch-geometric==1.7.2

#We got the file requirements.txt from here: https://github.com/nicolas-chaulet/torch-points3d/blob/master/requirements.txt but deleted the lines containing the following libraries in the file: torch, torchvision, torch-geometric,  torch-scatter, torch-sparse, numpy
pip install -r requirements.txt

pip install numba==0.55.1

conda install -c conda-forge hdbscan==0.8.27

conda install numpy-base==1.19.2

pip install joblib==1.1.0
```

### Example 2 of installation
Specs local computer: Ubuntu 22.04.1, 64-bit, CUDA version 11.3
```bash
conda create -n torchpoint3denv python=3.8
conda activate torchpoint3denv
conda install -c conda-forge gcc==9.4.0
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install numpy==1.19.5
mamba install libopenblas openblas
find ${CONDA_PREFIX}/include -name "cblas.h"
export CXX=g++
export MAX_JOBS=2;
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

#THE STEPS START FROM HERE ARE EXACT THE SAME AS EXAMPLE 1 
#CHECK IF TORCH AND MINKOWSKI ENGINE WORK AS EXPECTED: 
#...

```

Based on our experience, we would suggest build most of the packages from the source for a larger chance of succesful installation. Good luck to your installation!

# Data introduction

## NPM3D dataset with instance labels
Link for dataset used in original paper:
https://doi.org/10.5281/zenodo.8188390
You can also download the latest dataset with small labeling corrections here:
https://github.com/bxiang233/PanopticSegForLargeScalePointCloud.git


### Semantic labels
1: "ground",
2: "buildings",
3: "poles",
4: "bollards",
5: "trash cans",
6: "barriers",
7: "pedestrians",
8: "cars",
9: "natural"

### Data folder structure
```bash
├─ conf                    # All configurations for training and evaluation leave there
├─ forward_scripts         # Script that runs a forward pass on possibly non annotated data
├─ outputs                 # All outputs from your runs
├─ scripts                 # Some scripts to help manage the project
├─ torch_points3d
├─ data                    # DATA FOLDER
    └─ npm3dfused
        └─ raw
            ├─ Lille1_1.ply          
            ├─ Lille1_2.ply   
            ├─ Lille2.ply           
            └─ Paris.ply           
├─ train.py                # Main script to launch a training
└─ eval.py                 # Eval script
```

## S3DIS dataset
Download Stanford3dDataset_v1.2_Version.zip from: http://buildingparser.stanford.edu/dataset.html#Download
 and unzip it.

### Data folder structure
```bash
├─ conf                    # All configurations for training and evaluation leave there
├─ forward_scripts         # Script that runs a forward pass on possibly non annotated data
├─ outputs                 # All outputs from your runs
├─ scripts                 # Some scripts to help manage the project
├─ torch_points3d
├─ data                    # DATA FOLDER
    └─ s3disfused
        └─ raw
            ├─ Area1
                └─ $ROOMTYPE$_$ID$
            ├─ Area2
                └─ $ROOMTYPE$_$ID$
            ├─ Area3
                └─ $ROOMTYPE$_$ID$
            ├─ Area4
                └─ $ROOMTYPE$_$ID$
            ├─ Area5
                └─ $ROOMTYPE$_$ID$
            └─ Area6
                └─ $ROOMTYPE$_$ID$
├─ train.py                # Main script to launch a training
└─ eval.py                 # Eval script
```

# Getting started with code
1. Create wandb account and specify your own wandb account name in conf/training/*.yaml. Have a look at all needed configurations of your current run in conf/data/panoptic/*.yaml, conf/models/panoptic/*.yaml and conf/training/*.yaml. Perform training by running:

```bash
# Running KPConv on NPM3D dataset
python train.py task=panoptic data=panoptic/npm3d-kpconv models=panoptic/kpconv model_name=KPConvPaperNPM3D training=npm3d_benchmark/kpconv-panoptic

# Running KPConv on S3DIS dataset
python train.py task=panoptic data=panoptic/s3disfused-kpconv models=panoptic/kpconv model_name=KPConvPaperS3DIS training=s3dis_benchmark/kpconv-panoptic

# Running Sparse CNN on NPM3D dataset
python train.py task=panoptic data=panoptic/npm3d-sparseconv models=panoptic/minkowski model_name=MinkowskiBackboneNPM3D training=npm3d_benchmark/minkowski-panoptic

# Running Sparse CNN on S3DIS dataset
python train.py task=panoptic data=panoptic/s3disfused-sparseconv models=panoptic/minkowski model_name=MinkowskiBackboneS3DIS training=s3dis_benchmark/minkowski-panoptic

# Running PointNet++ on NPM3D dataset
python train.py task=panoptic data=panoptic/npm3d-pointnet2 models=panoptic/pointnet2 model_name=pointnet2NPM3D training=npm3d_benchmark/pointnet2-panoptic

# Running PointNet++ on S3DIS dataset
python train.py task=panoptic data=panoptic/s3disfused-pointnet2 models=panoptic/pointnet2 model_name=pointnet2_indoor training=s3dis_benchmark/pointnet2-panoptic
```

2. Perform test by running:
```bash
python eval.py
```

3. Get final evaluation metrics run:
```bash
# For NPM3D dataset
python eval_PanopticSeg_NPM3D.py

# For S3DIS dataset
python generate_seperateRoom_forS3DIStest.py
python eval_PanopticSeg_S3DIS.py
```

# Citing
If you find our work useful, please do not hesitate to cite our paper:

```
@article{
  Xiang2023Review,
  title={A Review of Panoptic Segmentation for Mobile Mapping Point Clouds},
  author={Binbin Xiang and Yuanwen Yue and Torben Peters and Konrad Schindler},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2023},
  url = {\url{https://arxiv.org/abs/2304.13980}}
}
```

If you use NPM3D dataset, please do not forget to also cite:
```
@article{roynard2017parislille3d,
  author = {Xavier Roynard and Jean-Emmanuel Deschaud and François Goulette},
  title ={Paris-Lille-3D: A large and high-quality ground-truth urban point cloud dataset for automatic segmentation and classification},
  journal = {The International Journal of Robotics Research},
  volume = {37},
  number = {6},
  pages = {545-557},
  year = {2018},
  doi = {10.1177/0278364918767506}
}
```

If you use S3DIS dataset, please do not forget to also cite:
```
@InProceedings{armeni_cvpr16,
	title ={3D Semantic Parsing of Large-Scale Indoor Spaces},
	author = {Iro Armeni and Ozan Sener and Amir R. Zamir and Helen Jiang and Ioannis Brilakis and Martin Fischer and Silvio Savarese},
	booktitle = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition},
	year = {2016}, }
```