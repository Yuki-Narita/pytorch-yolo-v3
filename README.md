# ROS-kinetic + ZED 環境ですぐに使えるようにしたリポジトリ


## ROS環境で使うためのチュートリアル
### このリポジトリのクローンと重みのダウンロード
```
$ git clone https://github.com/Yuki-Narita/pytorch-yolo-v3.git
$ cd pytorch-yolo-v3
$ wget https://pjreddie.com/media/files/yolov3.weights
```
### ライブラリのインストール(本当に全部必要かは不明)
共通
```
$ sudo apt install python3-pip python3-tk
$ pip3 install pandas opencv-python cython
$ pip3 install pyswarms 'matplotlib<3.0'
```
GPUなし環境
```
$ pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
$ pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl
```
GPU(CUDA9.0)環境 (未検証)
```
$ pip3 install torch torchvision
```
GPU(CUDA10.0)環境
```
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
$ pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl
```

### カメラを使ったリアルタイム画像認識
<!--
カメラ接続後パーミッションを変更(カメラが /dev/video0 に接続されている場合の例)
```
$ sudo chmod 777 /dev/video0
```
-->
yolo実行(pytorch-yolo-v3 内で)
```
$ python3 cam_demo_ZED.py
```
オプションを付ける場合の例(詳細は下の方の公式README)
```
$ python3 cam_demo_ZED.py --confidence 0.6 --reso 320
```
### メモ
* 下記の公式チュートリアルのコマンドを使う場合 python と書かれている部分は python3 に置き換えて実行する
* ZEDの解像度変更は cam_demo_ZED.py の cap.set(3, width) cap.set(4, height)
* カメラが /dev/video1 に接続されている場合は cam_demo_ZED.py の cap = cv2.VideoCapture(0) を cv2.VideoCapture(1) に変更

# A PyTorch implementation of a YOLO v3 Object Detector

[UPDATE] : This repo serves as a driver code for my research. I just graduated college, and am very busy looking for research internship / fellowship roles before eventually applying for a masters. I won't have the time to look into issues for the time being. Thank you.


This repository contains code for a object detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, by [marvis](https://github.com/marvis/pytorch-yolo2). One of the goals of this code is to improve
upon the original port by removing redundant parts of the code (The official code is basically a fully blown deep learning 
library, and includes stuff like sequence models, which are not used in YOLO). I've also tried to keep the code minimal, and 
document it as well as I can. 

### Tutorial for building this detector from scratch
If you want to understand how to implement this detector by yourself from scratch, then you can go through this very detailed 5-part tutorial series I wrote on Paperspace. Perfect for someone who wants to move from beginner to intermediate pytorch skills. 

[Implement YOLO v3 from scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

As of now, the code only contains the detection module, but you should expect the training module soon. :) 

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

Using PyTorch 0.3 will break the detector.



## Detection Example

![Detection Example](https://i.imgur.com/m2jwneng.png)
## Running the detector

### On single or multiple images

Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file
This time around, for v3, authors has supplied a weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights), and place 

the weights file into your repo directory. Or, you could just type (if you're on Linux)

```
wget https://pjreddie.com/media/files/yolov3.weights 
python detect.py --images imgs --det det 
```


`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size (using `--bs` flag) , object threshold confidence can be tweaked with flags that can be looked up with. 

```
python detect.py -h
```

### Speed Accuracy Tradeoff
You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. 

```
python detect.py --images imgs --det det --reso 320
```

### On Video
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .avi format
since openCV only accepts OpenCV as the input format. 

```
python video_demo.py --video video.avi
```

Tweakable settings can be seen with -h flag. 

### Speeding up Video Inference

To speed video inference, you can try using the video_demo_half.py file instead which does all the inference with 16-bit half 
precision floats instead of 32-bit float. I haven't seen big improvements, but I attribute that to having an older card 
(Tesla K80, Kepler arch). If you have one of cards with fast float16 support, try it out, and if possible, benchmark it. 

### On a Camera
Same as video module, but you don't have to specify the video file since feed will be taken from your camera. To be precise, 
feed will be taken from what the OpenCV, recognises as camera 0. The default image resolution is 160 here, though you can change it with `reso` flag.

```
python cam_demo.py
```
You can easily tweak the code to use different weightsfiles, available at [yolo website](https://pjreddie.com/darknet/yolo/)

NOTE: The scales features has been disabled for better refactoring.
### Detection across different scales
YOLO v3 makes detections across different scales, each of which deputise in detecting objects of different sizes depending upon whether they capture coarse features, fine grained features or something between. You can experiment with these scales by the `--scales` flag. 

```
python detect.py --scales 1,3
```


