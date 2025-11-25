# import warnings, os
# # os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# # os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# # 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!
# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!

# if __name__ == '__main__':
#     model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
#     # model.load('yolo11n.pt') # loading pretrain weights
#     model.train(data='/root/dataset/dataset_visdrone/data.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=300,
#                 batch=32,
#                 close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
#                 workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
#                 # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
#                 optimizer='SGD', # using SGD
#                 # patience=0, # set 0 to close earlystop.
#                 # resume=True, # 断点续训,YOLO初始化时选择last.pt
#                 # amp=False, # close amp | loss出现nan可以关闭amp
#                 # fraction=0.2,
#                 project='runs/train',
#                 name='exp',
#                 )

from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    model = YOLO(r'D:\py\cxr\ultralytics-yolo11-main\my_yolo11.yaml')  # build a new model from YAML
    model = YOLO(r'D:\py\cxr\ultralytics-yolo11-main\yolo11n.pt')  # load a pretrained model (recommended for training)
    model = YOLO(r'D:\py\cxr\ultralytics-yolo11-main\my_yolo11.yaml').load(r'D:\py\cxr\ultralytics-yolo11-main\yolo11n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r'D:\py\cxr\ultralytics-yolo11-main\ultralytics\cfg\datasets\my1_coco128.yaml',
                          epochs=100, imgsz=640, batch=12, workers=4, device='0', patience=False)

