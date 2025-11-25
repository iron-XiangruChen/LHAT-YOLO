# from ultralytics import YOLO
# model = YOLO(r'D:\py\cxr\ultralytics-yolo11-main\runs\detect\v11+GSConv+Detect_LSCD\weights\best.pt')
# print(model.model)   # 打印模型结构
import torch
torch.cuda.is_available()

torch.backends.cudnn.is_available()

#查看版本号

print(torch.version.cuda)

print(torch.backends.cudnn.version())