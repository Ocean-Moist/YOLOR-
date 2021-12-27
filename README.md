## YOLOR Training with cusstom dataset

![](figure/unifued_network.png)

Thanks to Wong Kin Yiu [WongKinYiu](https://github.com/WongKinYiu/yolor) His code is open source, also his Pytorch versions of YOLOv4 and Scaled-YOLOv4 are quite good!

+ GitHub: https://github.com/WongKinYiu/yolor

+ Paper: https://arxiv.org/abs/2105.04206

  

![](figure/performance.png)

We will tell you how to train your own YOLOR models based on custom datasets in the following sections, and provide a deployment solution for the C++ version of YOLOR based on TensorRT.

+ Install the YOLOR dependency environment
+ Create your own dataset for YOLOR target detection
+ Prepare YOLOR pre-trained models
+ YOLOR model training
+ Object detection with YOLOR models (image and video based)
+ C++ code implementation of YOLOR-based TensorRT

### 1. Install the YOLOR dependency environment

Uses Docker to create the YOLOR image, which is also the author's recommended way,  assumes that the reader has installed Docker and nvidia-docker2. Only works under Linux, because nvidia-docker2 only works under Linux, if you are a windows system is recommended that you install the following environment by way of virtual machine or directly under the windows host.

```shell
# create docker env with nvidia-docker2
docker pull nvcr.io/nvidia/pytorch:20.11-py3
nvidia-docker run --name yolor -it -v your_coco_path/:/coco/ -v your_code_path/:/yolor --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3
# sudo nvidia-docker run --name yolor -it --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

#  install required packages in countainer
apt update
apt install -y zip htop screen libgl1-mesa-glx

pip install seaborn thop
pip install nvidia-pyindex
pip install onnx-graphsurgeon


# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# install pytorch_wavelets if you want to use dwt down-sampling module
# https://github.com/fbcotter/pytorch_wavelets
cd /
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .

# go to code folderls
cd /yolor

```
### 2. Create your own YOLOR  dataset

YOLOR supports YOLOv5  annotated data, if you are familiar with the creation of YOLOv5 training set, this part can be skipped, here we provide the code to build the dataset, store the dataset in `. /datasets` under.

```shell
./datasets/score   
├─images           # Training images, with training and evaluation images stored under each folder
│  ├─train
│  └─val
└─labels           # label，each folder holds txt annotation files in YOLOv5 format
    ├─train
    └─val
```

Code for converting VOC annotation data format to YOLOv5 annotation, which is stored in `. /datasets`, for details on YOLOv5 annotation you can refer to: <https://github.com/DataXujing/YOLO-v5>


### 3. Prepare the pre-training model for YOLOR

+ 1. Modify the model's configuration file

i. YAML of training data `. /data/score.yaml`

```shell
train: ./datasets/score/images/train/
val: ./datasets/score/images/val/

# number of classes
nc: 3
# class names
names: ['QP', 'NY', 'QG']
```

ii. config of model structure 

We train YOLOR-P6, we need to modify the configuration file of the model, its modification is similar to the darkent version of YOLOv3, the main modified parameters in the head part of the model, detailed reference `. /cfg/yolor_p6_score.cfg`, whose main modification part is as follows.

```shell
# ============ End of Neck ============ #

# 203
[implicit_add]
filters=256

# 204
[implicit_add]
filters=384

# 205
[implicit_add]
filters=512

# 206
[implicit_add]
filters=640

# 207   #<------------(number_class + 5) *3
[implicit_mul]
filters=24      

# 208   #<------------(number_class + 5) *3
[implicit_mul]
filters=24

# 209  #<------------(number_class + 5) *3
[implicit_mul]
filters=24

# 210  #<------------(number_class + 5) *3
[implicit_mul]
filters=24

# ============ Head ============ #

# YOLO-3

[route]
layers = 163

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=silu

[shift_channels]
from=203

# <---------------- filters: (number_class + 5) *3
[convolutional]
size=1
stride=1
pad=1
filters=24
activation=linear

[control_channels]
from=207

# <---------------classess: 3
[yolo]
mask = 0,1,2
anchors = 19,27,  44,40,  38,94,  96,68,  86,152,  180,137,  140,301,  303,264,  238,542,  436,615,  739,380,  925,792
classes=3
num=12
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
scale_x_y = 1.05
iou_thresh=0.213
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
nms_kind=greedynms
beta_nms=0.6


# YOLO-4

[route]
layers = 176

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=384
activation=silu

[shift_channels]
from=204

# <---------------- filters: (number_class + 5) *3
[convolutional]
size=1
stride=1
pad=1
filters=24
activation=linear

[control_channels]
from=208

# <--------------- classes: 3
[yolo]
mask = 3,4,5
anchors = 19,27,  44,40,  38,94,  96,68,  86,152,  180,137,  140,301,  303,264,  238,542,  436,615,  739,380,  925,792
classes=3
num=12
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
scale_x_y = 1.05
iou_thresh=0.213
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
nms_kind=greedynms
beta_nms=0.6


# YOLO-5

[route]
layers = 189

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=silu

[shift_channels]
from=205

# <---------------- filters: (number_class + 5) *3
[convolutional]
size=1
stride=1
pad=1
filters=24
activation=linear

[control_channels]
from=209

# <------------------classes: 3
[yolo]
mask = 6,7,8
anchors = 19,27,  44,40,  38,94,  96,68,  86,152,  180,137,  140,301,  303,264,  238,542,  436,615,  739,380,  925,792
classes=3
num=12
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
scale_x_y = 1.05
iou_thresh=0.213
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
nms_kind=greedynms
beta_nms=0.6


# YOLO-6

[route]
layers = 202

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=640
activation=silu

[shift_channels]
from=206

# <---------------- filters: (number_class + 5) *3
[convolutional]
size=1
stride=1
pad=1
filters=24
activation=linear

[control_channels]
from=210

# <-------------classes： 3
[yolo]
mask = 9,10,11
anchors = 19,27,  44,40,  38,94,  96,68,  86,152,  180,137,  140,301,  303,264,  238,542,  436,615,  739,380,  925,792
classes=3
num=12
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
scale_x_y = 1.05
iou_thresh=0.213
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
nms_kind=greedynms
beta_nms=0.6

# ============ End of Head ============ #

```

+ 2. Download the pre-trained models

To reproduce the results in the paper, please use this branch.

| Model         | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>S</sub><sup>test</sup> | AP<sub>M</sub><sup>test</sup> | AP<sub>L</sub><sup>test</sup> | batch1 throughput |
| :------------ | :-------: | :---------------: | :----------------------------: | :----------------------------: | :---------------------------: | :---------------------------: | :---------------------------: | :---------------: |
| **YOLOR-P6**  |   1280    |     **52.6%**     |           **70.6%**            |           **57.6%**            |           **34.7%**           |           **56.6%**           |           **64.2%**           |     49 *fps*      |
| **YOLOR-W6**  |   1280    |     **54.1%**     |           **72.0%**            |           **59.2%**            |           **36.3%**           |           **57.9%**           |           **66.1%**           |     47 *fps*      |
| **YOLOR-E6**  |   1280    |     **54.8%**     |           **72.7%**            |           **60.0%**            |           **36.9%**           |           **58.7%**           |           **66.9%**           |     37 *fps*      |
| **YOLOR-D6**  |   1280    |     **55.4%**     |           **73.3%**            |           **60.6%**            |           **38.0%**           |           **59.2%**           |           **67.1%**           |     30 *fps*      |
|               |           |                   |                                |                                |                               |                               |                               |                   |
| **YOLOv4-P5** |    896    |     **51.8%**     |           **70.3%**            |           **56.6%**            |           **33.4%**           |           **55.7%**           |           **63.4%**           |     41 *fps*      |
| **YOLOv4-P6** |   1280    |     **54.5%**     |           **72.6%**            |           **59.8%**            |           **36.6%**           |           **58.2%**           |           **65.5%**           |     30 *fps*      |
| **YOLOv4-P7** |   1536    |     **55.5%**     |           **73.4%**            |           **60.8%**            |           **38.4%**           |           **59.4%**           |           **67.7%**           |     16 *fps*      |
|               |           |                   |                                |                                |                               |                               |                               |                   |

| Model                                 | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | AP<sub>S</sub><sup>val</sup> | AP<sub>M</sub><sup>val</sup> | AP<sub>L</sub><sup>val</sup> | FLOPs |                           weights                            |
| :------------------------------------ | :-------: | :--------------: | :---------------------------: | :---------------------------: | :--------------------------: | :--------------------------: | :--------------------------: | :---: | :----------------------------------------------------------: |
| **YOLOR-P6**                          |   1280    |    **52.5%**     |           **70.6%**           |           **57.4%**           |          **37.4%**           |          **57.3%**           |          **65.2%**           | 326G  | [yolor-p6.pt](https://drive.google.com/file/d/1WyzcN1-I0n8BoeRhi_xVt8C5msqdx_7k/view?usp=sharing) |
| **YOLOR-W6**                          |   1280    |    **54.0%**     |           **72.1%**           |           **59.1%**           |          **38.1%**           |          **58.8%**           |          **67.0%**           | 454G  | [yolor-w6.pt](https://drive.google.com/file/d/1KnkBzNxATKK8AiDXrW_qF-vRNOsICV0B/view?usp=sharing) |
| **YOLOR-E6**                          |   1280    |    **54.6%**     |           **72.5%**           |           **59.8%**           |          **39.9%**           |          **59.0%**           |          **67.9%**           | 684G  | [yolor-e6.pt](https://drive.google.com/file/d/1jVrq8R1TA60XTUEqqljxAPlt0M_MAGC8/view?usp=sharing) |
| **YOLOR-D6**                          |   1280    |    **55.4%**     |           **73.5%**           |           **60.6%**           |          **40.4%**           |          **60.1%**           |          **68.7%**           | 937G  | [yolor-d6.pt](https://drive.google.com/file/d/1WX33ymg_XJLUJdoSf5oUYGHAtpSG2gj8/view?usp=sharing) |
|                                       |           |                  |                               |                               |                              |                              |                              |       |                                                              |
| **YOLOR-S**                           |    640    |    **40.7%**     |           **59.8%**           |           **44.2%**           |          **24.3%**           |          **45.7%**           |          **53.6%**           |  21G  |                                                              |
| **YOLOR-S**<sub>DWT</sub>             |    640    |    **40.6%**     |           **59.4%**           |           **43.8%**           |          **23.4%**           |          **45.8%**           |          **53.4%**           |  21G  |                                                              |
| **YOLOR-S<sup>2</sup>**<sub>DWT</sub> |    640    |    **39.9%**     |           **58.7%**           |           **43.3%**           |          **21.7%**           |          **44.9%**           |          **53.4%**           |  20G  |                                                              |
| **YOLOR-S<sup>3</sup>**<sub>S2D</sub> |    640    |    **39.3%**     |           **58.2%**           |           **42.4%**           |          **21.3%**           |          **44.6%**           |          **52.6%**           |  18G  |                                                              |
| **YOLOR-S<sup>3</sup>**<sub>DWT</sub> |    640    |    **39.4%**     |           **58.3%**           |           **42.5%**           |          **21.7%**           |          **44.3%**           |          **53.0%**           |  18G  |                                                              |
| **YOLOR-S<sup>4</sup>**<sub>S2D</sub> |    640    |    **36.9%**     |           **55.3%**           |           **39.7%**           |          **18.1%**           |          **41.9%**           |          **50.4%**           |  16G  | [weights](https://drive.google.com/file/d/1rFoRk1ZoKvE8kbxAl2ABBy6m9Zl6_k4Y/view?usp=sharing) |
| **YOLOR-S<sup>4</sup>**<sub>DWT</sub> |    640    |    **37.0%**     |           **55.3%**           |           **39.9%**           |          **18.4%**           |          **41.9%**           |          **51.0%**           |  16G  | [weights](https://drive.google.com/file/d/1IZ1ix1hwUjEMcCMpl67CHQp98XOjcIsv/view?usp=sharing) |
|                                       |           |                  |                               |                               |                              |                              |                              |       |                                                              |

It should be noted that the pre-training model download address in the above table corresponds to the results in the author's paper, which cannot be loaded in the training of this project, if you use this project need to load the pre-training model during training download the following link.

```shell
# YOLOR-P6:
https://drive.google.com/uc?export=download&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76

# YOLOR-W6
https://drive.google.com/uc?export=download&id=1UflcHlN5ERPdhahMivQYCbWWw7d2wY7U

#YOLOR-CSP
https://drive.google.com/file/d/1ZEqGy4kmZyD-Cj3tEFJcLSZenZBDGiyg/view?usp=sharing

# YOLOR-CSP*
https://drive.google.com/file/d/1OJKgIasELZYxkIjFoiqyn555bcmixUP2/view?usp=sharing

# YOLOR-CSP-X
https://drive.google.com/file/d/1L29rfIPNH1n910qQClGftknWpTBgAv6c/view?usp=sharing

# YOLOR-CSP-X*
https://drive.google.com/file/d/1NbMG3ivuBQ4S8kEhFJ0FIqOQXevGje_w/view?usp=sharing
```



### 4. Model training

The main params：

- **img:** iamge px size
- **batch:** batch size
- **epochs:** cycle time 
- **data:** yaml config file path
- **cfg:** model profile
- **weights:** path for pre-trained model weights
- **name:** result names
- **hyp:** hyper params

```shell
python train.py --batch-size 8 --img 1280 1280 --data './data/score.yaml' --cfg cfg/yolor_p6_score.cfg --weights './pretrain/yolor-p6.pt' --device 0 --name yolor_p6 --hyp './data/hyp.scratch.1280.yaml' --epochs 300
# To verify the feasibility of the process, we trained only 50 epochs!!!
```

view training progress：

```shell
tensorboard --logdir "./yolor_p6" --host 0.0.0.0
```

![](figure/tensorboard.png)

Results of training：

![](figure/results.png)

### 5. object detection

To facilitate testing and deployment, based on `detect.py` we implemented the test code for images and videos, which are stored in `test_img.py` and `test_video.py`, respectively, and are called as follow

```shell
# image object detection
python test_img.py

# video object detection
python test_video.py
```

demo:

![](figure/test_res.jpg)



### 6.TensorRT C++ impelmentation

1.Convert model to ONNX

```shell
  python convert_to_onnx.py --weights ./runs/train/yolor_p6/weights/best_overall.pt --cfg cfg/yolor_p6_score.cfg --output yolor_p6.onnx
```

**TODO**: The SILU activation function cannot be converted to ONNX in Pytorch  will spend time to solve the problem later.
