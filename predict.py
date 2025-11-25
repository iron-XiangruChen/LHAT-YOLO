from ultralytics import YOLO

#model = YOLO(r'D:\\py\\cxr\\Shale-Yolo\\runs1\detect\\train7(shale-yolo)\\weights\\best.pt')
model = YOLO(r'D:\py\cxr\ultralytics-yolo11-main\runs\detect\V11-100\weights\best.pt')
model.predict(source=r'E:\实验室资料\论文撰写\深度学习-施工现场\图像与数据\测试\原图', conf=0.05, save=True, show_conf=False, save_conf=True, save_txt=True, name=r'E:\实验室资料\论文撰写\深度学习-施工现场\图像与数据\测试\结果')