# Semi-supervised-surgical-tool-detection
This repository contains code for our paper titled "A semi-supervised teacher-student framework for surgical tool detection and localization"
### Dataset Download 
m2cai16-tool locations dataset can be downloaded [here](https://ai.stanford.edu/~syyeung/tooldetection.html)
Dataset annotations are in VOC format. However, this work uses coco format. All the required code files for voc to coco conversion can be found in data folder. 

### Checkpoints
Download checkpoint [here](https://drive.google.com/file/d/1CrS4oKPWZAlAJh1m1NzyuB4019r_-GvP/view?usp=sharing)

### Build Environment 
`# create conda env` <br />
`conda create -n ut python=3.6`<br />
`# activate the enviorment` <br />
`conda activate ut` <br />
`# install PyTorch >=1.5 with GPU` <br />
`conda install pytorch torchvision -c pytorch` <br />
`# install detectron2` <br />
[link](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) to install <br />

### Training 
* To train the network use following. <br />
`CUDA_VISIBLE_DEVICES =1,2 python train_net.py` \ <br /> `--num-gpus 2` \ <br />`--config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml\`<br />`SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4` <br />

* Just change the config file to train on different percentages of labeled set. <br />

### Evaluation
* To evaluate the model, use the checkpoint. <br />
`CUDA_VISIBLE_DEVICES =1,2 python train_net.py` \ <br />  `--eval-only` \ <br />  `--num-gpus 2` \ <br />`--config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml\`<br />`SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4` \ <br /> 
`MODEL.WEIGHTS path_to_checkpoint/checkpoint` 
