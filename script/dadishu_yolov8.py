from PIL import ImageGrab
import time
import pyautogui
from ultralytics import YOLO

# 加载 YOLOv8 自定义模型，请确保 'yolov8_best.onnx' 为你训练好的权重文件路径
model = YOLO('./yolov8_best.onnx')

# 循环执行（此处循环 10000 次，可根据需要修改）
for i in range(10000):
    # 截取屏幕指定区域（根据实际情况调整 bbox 参数）
    img = ImageGrab.grab(bbox=(0, 0, 500, 1080))
    
    # 使用 YOLOv8 进行目标检测，并设置置信度阈值为 0.3
    results = model(img, conf=0.3)
    
    # 由于只传入了一张图片，因此取第一个检测结果
    result = results[0]
    
    # 遍历检测到的所有目标
    for box in result.boxes:
        # 获取目标的边界框坐标：[xmin, ymin, xmax, ymax]
        xyxy = box.xyxy.cpu().numpy().flatten()
        xmin, ymin, xmax, ymax = xyxy
        
        # 获取目标类别索引（转换为整数）
        cls = int(box.cls.cpu().numpy()[0])
        # 从模型中获取类别名称（确保模型的类别名称与训练时一致）
        label = model.names[cls]
        
        # 如果检测到的目标为 "dishu"（打地鼠目标）
        if label == "dishu":
            # 计算目标中心点坐标
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            print(f"点击坐标：({x_center}, {y_center})")
            
            # 模拟鼠标点击
            pyautogui.click(x_center, y_center)
    
    # 暂停 0.05 秒后继续下一次循环
    time.sleep(0.05)