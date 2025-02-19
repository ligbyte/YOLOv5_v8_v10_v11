from PIL import ImageGrab
import time
import torch
import json
import pyautogui


img = ImageGrab.grab(bbox=(0,0,600,1080))
img.save('test.png')