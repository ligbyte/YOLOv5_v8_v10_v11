from PIL import ImageGrab
import time
import torch
import json
import pyautogui
#导入yolov5目录  和best权重
model = torch.hub.load('./yolov5', 'custom', path='./best.onnx', source='local')  # local repo
model.conf = 0.3  # 相似度

#循环截图
for i in range(10000):
    img = ImageGrab.grab(bbox=(0,0,500,1080))
    # img.save('test.png')
    #识别
    results=model(img)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    results= json.loads(detect_res)
    for result in results:
        if result["name"] =="dishu":
            x_center = (result["xmin"] + result["xmax"]) / 2
            y_center = (result["ymin"] + result["ymax"]) / 2
            print(x_center,y_center)
            #点击
            pyautogui.click(x_center, y_center)
    time.sleep(0.05)