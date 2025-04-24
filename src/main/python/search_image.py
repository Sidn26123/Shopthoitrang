# import math
# import os

# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import  Model

# from PIL import Image
# import pickle
# import numpy as np

import os
import sys
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')
root_path = os.path.dirname(os.path.abspath(__file__)) + "\\"
# Hàm tạo model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Hàm tiền xử lý, chuyển đổi hình ảnh thành tensor
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Xu ly:", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Trích đặc trưng
    vector = model.predict(img_tensor)[0]
    # Chuẩn hóa vector = chia L2 norm
    vector = vector / np.linalg.norm(vector)
    return vector

# Định nghĩa ảnh cần tìm kiếm
print("B")
search_image = str(sys.argv[1])

# Khởi tạo model
model = get_extract_model()

# Trích đặc trưng ảnh search
search_vector = extract_vector(model, search_image)

# Load 4700 vector từ vectors.pkl ra biến
vectors = pickle.load(open(root_path+"vectors.pkl", "rb"))
paths = pickle.load(open(root_path+"paths.pkl", "rb"))

# Tính khoảng cách từ search_vector đến tất cả các vector
distance = np.linalg.norm(vectors - search_vector, axis=1)

# Sắp xếp và lấy ra K vector có khoảng cách ngắn nhất
K = 16
ids = np.argsort(distance)[:K]

nearest_image_names = []
for id in ids:
    if distance[id] <= 0.66:
        image_path = paths[id]
        image_name = os.path.basename(image_path)  # Lấy tên tệp ảnh từ đường dẫn
        image_name_without_extension = os.path.splitext(image_name)[0]  # Loại bỏ phần mở rộng ".jpg"
        nearest_image_names.append(image_name_without_extension)


# Trả về danh sách các tên tệp ảnh phù hợp
for name in nearest_image_names:
    print("name:", name)

# import os
# import sys
# import torch
# import pickle
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch.nn as nn

# # Set UTF-8 encoding for stdout
# sys.stdout.reconfigure(encoding='utf-8')

# def get_extract_model():
#     model = models.vgg16(pretrained=True)
#     model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
#     model.eval()
#     return model

# def image_preprocess(img):
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = preprocess(img).unsqueeze(0)
#     return img_tensor

# def extract_vector(model, image_path):
#     print("Xu ly:", image_path)
#     img = Image.open(image_path).convert("RGB")
#     img_tensor = image_preprocess(img)

#     # Trích đặc trưng
#     with torch.no_grad():
#         vector = model(img_tensor).numpy()[0]
#     # Chuẩn hóa vector = chia L2 norm
#     vector = vector / np.linalg.norm(vector)
#     return vector

# # Định nghĩa ảnh cần tìm kiếm
# search_image = str(sys.argv[1])

# # Khởi tạo model
# model = get_extract_model()

# # Trích đặc trưng ảnh search
# search_vector = extract_vector(model, search_image)

# # Load 4700 vector từ vectors.pkl ra biến
# vectors = pickle.load(open("vectors.pkl", "rb"))
# paths = pickle.load(open("paths.pkl", "rb"))

# # Tính khoảng cách từ search_vector đến tất cả các vector
# distance = np.linalg.norm(vectors - search_vector, axis=1)

# # Sắp xếp và lấy ra K vector có khoảng cách ngắn nhất
# K = 16
# ids = np.argsort(distance)[:K]

# nearest_image_names = []
# for id in ids:
#     if distance[id] <= 0.66:
#         image_path = paths[id]
#         image_name = os.path.basename(image_path)  # Lấy tên tệp ảnh từ đường dẫn
#         image_name_without_extension = os.path.splitext(image_name)[0]  # Loại bỏ phần mở rộng ".jpg"
#         nearest_image_names.append(image_name_without_extension)

# # Trả về danh sách các tên tệp ảnh phù hợp
# for name in nearest_image_names:
#     print("name:", name)
