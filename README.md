# 安装

1 [官网最新版下载连接](https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe)


2 [通过conda搭建环境](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)


安装之后就能使用python和pip

3. [git客户端安装](https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.1/Git-2.47.0-64-bit.exe)



4. 安装yolo环境

官网教程：

<https://docs.ultralytics.com/quickstart/#install-ultralytics>

命令：
```
pip install ultralytics
```
5. 测试  python-3.12.4
```
yolo train data=datasets/coco128/coco128.yaml model=datasets/yolov8s.pt epochs=10 
```


CPU训练


```
yolo train data=D:\yolo\yolostudy\datasets\coco128\coco128.yaml model=datasets\yolov8s.pt epochs=10 
```

1. 云训练 python-3.10.0

<https://lanyun.net/term.html>

注册登录后，现在会送50元代金券，可以用来搭建训练服务器

云训练环境搭建教程，yolov5版本，这个版本可以在大漠工具中使用，方便测试效果

<https://docs.ultralytics.com/zh/yolov5/environments/aws_quickstart_tutorial/#step-3-connect-to-your-instance>

这里从本地上传后执行命令
```
unzip yolov5.zip
cd yolov5
pip install -r requirements.txt
```

模型下载地址

<https://github.com/ultralytics/yolov5/releases>

这里用v5s模型
将模型和图片数据上传到服务器中
```
cd ~
unzip datasets.zip
```
修改一下配置文件中的目录地址
path: /root/datasets/coco128




GPU训练命令
```
cd ~/yolov5

rm -rfv runs

#coco数据集
python train.py --patience 100  --weights  /home/lime/AI/yolo/yolov5/datasets/yolov5s.pt --data /home/lime/AI/yolo/yolov5/datasets/coco128/coco128.yaml --epochs 2001 --img 640

#打地鼠数据集
python train.py --patience 100  --weights  /home/lime/AI/yolo/yolov5/datasets/yolov5s.pt --data /home/lime/AI/yolo/yolov5/datasets/dadishu/dadishu.yaml --epochs 2001 --img 640

```
训练后的权重在runs中

GPU推测命令
```

#coco数据集
python detect.py --weights runs/train/exp/weights/best.pt --source  /home/lime/AI/yolo/yolov5/datasets/coco128/images/train2017


#打地鼠数据集
python detect.py --weights runs/train/exp/weights/best.pt --source  /home/lime/AI/yolo/yolov5/datasets/dadishu/images

```

用自己训练的权重去检测图片，效果很好

导出命令

```
python export.py --weights runs/train/exp/weights/best.pt --include onnx engine --img 640 --device 0 --dynamic
```

将训练的权重导出成onnx格式，方便其他程序使用
导出之前需要安装一个onnx库
pip install onnx onnxruntime

测试下在本地的效果
测试工具就用大漠工具，他需要的模型格式为onnx



7. 实战一下

https://h5.gaoshouyou.com/h5_game/dds/index.html

检测地鼠，然后控制鼠标去打
需要准备图片素材，用大漠工具获得，但是不用大漠训练，因为本机没有gpu速度慢。
将大漠标记好的标签文件夹复制到datasets中，压缩上传

需要改改数据路径就可以用
训练
```
cd ~/yolov5
python train.py --weights  ../datasets/yolov5s.pt --data ../datasets/dadishu/dadishu.yaml --epochs 50 --img 640

python train.py --weights  /home/lime/AI/yolo/yolov5/datasets/yolov5s.pt --data /home/lime/AI/yolo/yolov5/datasets/dadishu/dadishu.yaml --epochs 2001 --img 640
```
推测命令
```
python detect.py --weights runs/train/exp/weights/best.pt --source  /home/lime/AI/yolo/yolov5/datasets/dadishu/images/

导出命令
```
python export.py --weights runs/train/exp/weights/best.pt --include onnx engine --img 640 --device 0  --dynamic
```
效果very good

8. 写个简单脚本
先安装 
```
pip3 install pyautogui  onnx  gitpython onnxruntime ultralytics ipython

```
