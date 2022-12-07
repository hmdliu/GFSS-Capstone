# GFSS-Capstone

## Intro
This is a capstone project on generalized few-shot segmentation (GFSS). This repo presents a neat and scalable codebase for GFSS research. The code for our key contributions can be found in the following files:
- [Main Model](https://github.com/hmdliu/GFSS-Capstone/blob/main/src/model/meta/mib.py)
- [Contrastive Pre-training](https://github.com/hmdliu/GFSS-Capstone/blob/main/src/contrast.py)
- [Background Modeling Losses](https://github.com/hmdliu/GFSS-Capstone/blob/main/src/model/utils.py)

## Requisites
- Test Env: Python 3.9.7 (Singularity)
    - Path on NYU Greene: `/scratch/hl3797/overlay-25GB-500K.ext3`
- Packages:
    - torch (1.10.2+cu113), torchvision (0.11.3+cu113), timm (0.5.4)
    - numpy, scipy, pandas, tensorboardX
    - cv2, einops

## Clone codebase
```shell
git clone https://github.com/hmdliu/GFSS-Capstone && cd GFSS-Capstone
```

## Preparation

### PASCAL-5i dataset
**Note:** Make sure the path in `scripts/prepare_pascal.sh` works for you.
```shell
# default data root: ../dataset/VOCdevkit/VOC2012
bash scripts/prepare_pascal.sh
```

### COCO-20i dataset
You may refer to <a href="https://github.com/dvlab-research/PFENet#datasets-and-data-preparation" target="_blank">PFENet</a> for more details.

### Pretrained models
For ImageNet pre-trained weights, please download it <a href="https://drive.google.com/file/d/1rMPedZBKFXiWwRX3OHttvKuD1h9QRDbU/view?usp=sharing" target="_blank">here</a> (credits <a href="https://github.com/dvlab-research/PFENet#run-demo--test-with-pretrained-models" target="_blank">PFENet</a>) and unzip as `initmodel/`. \
For base class pre-trained weights, you may find it <a href="https://drive.google.com/file/d/1VPBquiy4DZXu8b6qsSB6XtO5_6jTpXgo/view?usp=sharing" target="_blank">here</a> (credits <a href="https://github.com/zhiheLu/CWT-for-FSS#pre-trained-models-in-the-first-stage" target="_blank">CWT</a>) and rename them as follows: `pretrained/[dataset]/split[i]/pspnet_resnet[layers]/best.pth`. We'll release our weights shortly.

## Dir explanations
- **initmodel**: ImageNet pre-trained backbone weights. `.pth`
- **pretrained**: Base classes pre-trained backbone weights. `.pth`
- **configs**: Base configurations for experiments. `.yaml`
- **scripts**: Training and helper scripts. `.sh` `.slurm`
- **results**: Logs and checkpoints. `.log` `.pth` `.yaml`
- **src**: Source code. `.py`

## Sample Usage
`exp_id` aims to make efficient config modifications for experiment purposes. It follows the format of `[exp_group]_[meta_cfg]_[train_cfg]`, see `src/exp.py` for a sample usage (pascal 1-shot on fold 0).
```shell
# debug mode (i.e., only log to shell)
python -m src.test --config configs/pascal_mib.yaml --exp_id sample_t2_pm10 --debug True

# submit to slurm
sbatch scripts/test_pascal.slurm configs/pascal_mib.yaml sample_t2_pm10

# output dir: results/sample/t2_pm10
tail results/sample/t2_pm10/output.log
```

## Credits
Major References: <a href="https://github.com/dvlab-research/PFENet" target="_blank">**PFENet**</a>, <a href="https://github.com/zhiheLu/CWT-for-FSS" target="_blank">**CWT**</a>, <a href="https://github.com/rstrudel/segmenter" target="_blank">**Segmenter**</a>, <a href="https://github.com/fcdl94/MiB" target="_blank">**MiB**</a>, and <a href="https://github.com/tfzhou/ContrastiveSeg" target="_blank">**ContrastiveSeg**</a>.

## Group Members
- Haoming Liu (hl3797@nyu.edu)
- Chengyu Zhang (cz1627@nyu.edu)
- Xiaochen Lu (xl3139@nyu.edu)

## Acknowledgements
We thank Professor Li Guo for her consistent guidance throughout the project. We thank Professor Hongyi Wen for his suggestions on the project write-up. This work was supported through the NYU IT High Performance Computing
resources, services, and staff expertise.