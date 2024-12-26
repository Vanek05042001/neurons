# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:27:02 2024

@author: Vanya
"""
import os
from glob import glob
import shutil
import ultralytics
from ultralytics import YOLO
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def create_yolo_structure(base_dir, class_names):
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        # Создаем директории images и labels
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            image_paths = glob(os.path.join(class_dir, '*.jpg'))
            
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                
                # Перемещаем изображение в images
                shutil.copy(image_path, os.path.join(images_dir, image_name))
                
                # Создаем метку в формате YOLO
                label_path = os.path.join(labels_dir, label_name)
                with open(label_path, 'w') as label_file:
                    # Вся картинка выделена под один класс
                    label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Папка с исходной структурой
base_dir = 'D:/Учёба/Магистратура/Нейронные сети/Лаб5/animals'
# Список классов с их названиями
class_names = ['cat', 'dog', 'wild']

# Создание структуры данных для YOLO
create_yolo_structure(base_dir, class_names)



ultralytics.checks()

# Загрузка классификационной модели
model = YOLO('D:/Учёба/Магистратура/Нейронные сети/Лаб5/animals_classifier/train12/weights/best.pt')
# model = YOLO("yolov8s.pt")

# Обучение модели
model.train(data='D:\\Учёба\\Магистратура\\Нейронные сети\\Лаб5\\animals.yaml', model="yolov8s.pt", epochs=1, imgsz=224, batch=16, 
            project='animals_classifier', val = True, verbose=True)


results = model("animals\\val\\images\\pixabay_dog_002239.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_cat_002662.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


import matplotlib.pyplot as plt
plt.imshow(result.plot())

results = model("animals\\val\\images\\pixabay_wild_000838.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("Снимок экрана 2022-08-08 220228.png")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_cat_001196.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_wild_001094.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_dog_003914.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_cat_001599.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_wild_001058.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())