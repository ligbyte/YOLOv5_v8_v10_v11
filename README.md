# YOLOv5
1 [YOLOv5 Github地址](https://github.com/ultralytics/yolov5)

2 [python官网最新版下载连接](https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe)


3 [通过conda搭建环境](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)


安装之后就能使用python和pip

1. [git客户端安装](https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.1/Git-2.47.0-64-bit.exe)



4. 安装yolo环境

官网教程：

<https://docs.ultralytics.com/quickstart/#install-ultralytics>

命令：
```shell
pip install ultralytics
```
5. 测试  python-3.12.4
```shell
yolo train data=datasets/coco128/coco128.yaml model=datasets/yolov8s.pt epochs=10 
```


## CPU训练


```shell
yolo train data=D:\yolo\yolostudy\datasets\coco128\coco128.yaml model=datasets\yolov8s.pt epochs=10 
```

1. 云训练 python-3.10.0

<https://lanyun.net/term.html>

注册登录后，现在会送50元代金券，可以用来搭建训练服务器

云训练环境搭建教程，yolov5版本，这个版本可以在大漠工具中使用，方便测试效果

<https://docs.ultralytics.com/zh/yolov5/environments/aws_quickstart_tutorial/#step-3-connect-to-your-instance>

这里从本地上传后执行命令
```shell
unzip yolov5.zip
cd yolov5
pip install -r requirements.txt
```

模型下载地址

<https://github.com/ultralytics/yolov5/releases>

这里用v5s模型
将模型和图片数据上传到服务器中
```shell
cd ~
unzip datasets.zip
```
修改一下配置文件中的目录地址
path: /root/datasets/coco128




## GPU训练命令
```shell
cd ~/yolov5

rm -rfv runs

//coco数据集
python train.py --patience 100  --weights  /home/lime/AI/yolo/yolov5/datasets/yolov5s.pt --data /home/lime/AI/yolo/yolov5/datasets/coco128/coco128.yaml --epochs 2001 --img 640

//打地鼠数据集
python train.py --patience 100  --weights  /home/lime/AI/yolo/yolov5/datasets/yolov5s.pt --data /home/lime/AI/yolo/yolov5/datasets/dadishu/dadishu.yaml --epochs 2001 --img 640

```

训练后的权重在runs中

## GPU推测命令

```shell
//coco数据集
python detect.py --weights runs/train/exp/weights/best.pt --source  /home/lime/AI/yolo/yolov5/datasets/coco128/images/train2017


//打地鼠数据集
python detect.py --weights runs/train/exp/weights/best.pt --source  /home/lime/AI/yolo/yolov5/datasets/dadishu/images

```

用自己训练的权重去检测图片，效果很好

## 导出命令

```shell
python export.py --weights runs/train/exp/weights/best.pt --include onnx engine --img 640 --device 0 --dynamic
```

将训练的权重导出成onnx格式，方便其他程序使用
导出之前需要安装一个onnx库
pip install onnx onnxruntime

测试下在本地的效果
测试工具就用大漠工具，他需要的模型格式为onnx



## 实战一下

https://h5.gaoshouyou.com/h5_game/dds/index.html

检测地鼠，然后控制鼠标去打
需要准备图片素材，用大漠工具获得，但是不用大漠训练，因为本机没有gpu速度慢。
将大漠标记好的标签文件夹复制到datasets中，压缩上传

需要改改数据路径就可以用
训练
```shell
cd ~/yolov5
python train.py --weights  ../datasets/yolov5s.pt --data ../datasets/dadishu/dadishu.yaml --epochs 50 --img 640

python train.py --weights  /home/lime/AI/yolo/yolov5/datasets/yolov5s.pt --data /home/lime/AI/yolo/yolov5/datasets/dadishu/dadishu.yaml --epochs 2001 --img 640
```
推测命令

```shell
python detect.py --weights runs/train/exp/weights/best.pt --source  /home/lime/AI/yolo/yolov5/datasets/dadishu/images/
```

导出命令
```shell
python export.py --weights runs/train/exp/weights/best.pt --include onnx engine --img 640 --device 0  --dynamic
```
效果very good

8. 写个简单脚本
先安装 

```shell
pip3 install pyautogui  onnx  gitpython onnxruntime ultralytics ipython

```




# YOLOv8
1 [YOLOv8 Github地址](https://github.com/ultralytics/yolov5)


# YOLOv10

1 [YOLOv10 Github地址](https://github.com/THU-MIG/yolov10)

## 开始配置python虚拟环境

code /home/lime/.bashrc

```shell

#Anaconda
# .bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!

__conda_setup="$('/home/lime/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/lime/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/lime/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/lime/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup


# <<< conda initialize <<<

```



```shell
cd /home/lime/AI/yolo/yolov10

conda create -n yolov10 python=3.9
conda activate yolov10

git clone git@github.com:THU-MIG/yolov10.git

cd yolov10
pip install -r requirements.txt
pip install -e .

```


## GPU训练命令

```shell

rm -rfv /home/lime/AI/yolo/yolov10/yolov10/runs

//coco数据集
yolo detect train data=/home/lime/AI/yolo/yolov10/datasets/coco128/coco128.yaml model=yolov10s.yaml epochs=2000 batch=32 imgsz=640 device=0

//打地鼠数据集
yolo detect train data=/home/lime/AI/yolo/yolov10/datasets/dadishu/dadishu.yaml model=yolov10s.yaml epochs=2000 batch=32 imgsz=640 device=0

```

训练后的权重在runs中


## GPU验证命令

```shell
//coco数据集
yolo val model=runs/detect/train/weights/best.pt data=/home/lime/AI/yolo/yolov10/datasets/coco128/coco128.yaml batch=32

//打地鼠数据集
yolo val model=runs/detect/train/weights/best.pt data=/home/lime/AI/yolo/yolov10/datasets/dadishu/dadishu.yaml batch=32

```


## GPU推理命令

```shell
//coco数据集
yolo predict model=runs/detect/train/weights/best.pt data=/home/lime/AI/yolo/yolov10/datasets/coco128/coco128.yaml source=/home/lime/AI/yolo/yolov10/datasets/coco128/images/train2017 batch=32


//打地鼠数据集
yolo predict model=runs/detect/train/weights/best.pt data=/home/lime/AI/yolo/yolov10/datasets/dadishu/dadishu.yaml source=/home/lime/AI/yolo/yolov10/datasets/dadishu/images/ batch=32

```

用自己训练的权重去检测图片，效果很好

## 导出命令

```shell
# End-to-End ONNX
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=13 simplify
# Predict with ONNX
yolo predict model=runs/detect/train/weights/best.onnx

# End-to-End TensorRT
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=engine half=True simplify opset=13 workspace=16
# or
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# Predict with TensorRT
yolo predict model=yolov10n/s/m/b/l/x.engine
```

将训练的权重导出成onnx格式，方便其他程序使用
导出之前需要安装一个onnx库

```shell
pip install onnx onnxruntime
```

测试下在本地的效果
测试工具就用大漠工具，他需要的模型格式为onnx

# YOLOv11
1 [YOLOv11 Github地址](https://github.com/ultralytics/yolov5)


# X-AnyLabeling GPU版本编译
Please compile the GPU version based on your specific CUDA version and hardware environment, as it relies on designated CUDA versions. You can find detailed compilation instructions here: https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/zh_cn/get_started.md