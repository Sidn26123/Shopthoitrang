import pandas as pd
import numpy as np
import sklearn
import sys
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine
import numpy as np
import os
from recommendTrain import get_model,map_item_to_id
import csv
root_path = os.path.dirname(os.path.abspath(__file__)) + "\\"
X = pd.read_csv(root_path+'ratings.csv', sep="\t", names =["mand", "masp", "rating"])
commender_products = pd.read_csv(root_path+'recommended_products.csv', sep="\t", names =["mand", "masp", "rating"])

mapped_ratings = X.copy()
mapped_items, item2id, id2item = map_item_to_id(mapped_ratings)
mapped_ratings['masp'] = X['masp'].map(item2id)
num_users = mapped_ratings["mand"].nunique()
num_items = mapped_ratings["masp"].nunique()
items = mapped_ratings["masp"].unique()
weight_path = root_path + "model_weights.weights.h5"
input_user_id = int(sys.argv[1])
model = get_model(num_users, num_items)
model.load_weights(weight_path, skip_mismatch=True)

#predictions = model.predict(input_product_str)
users = np.full(len(items), input_user_id, dtype='int32')
predictions = model.predict([users, np.array(items)], batch_size=100, verbose=0)
predictions = predictions.flatten()
recommended_items = [items[i] for i in np.argsort(predictions)[-10:][::-1]]
id2item_real= {}
with open(root_path+ 'recommended_products.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua tiêu đề cột
    for row in reader:
        id = int(row[0])  # ID sản phẩm
        product_name = row[1]  # Tên sản phẩm
        id2item_real[id] = product_name
for recommended_product in recommended_items:
    # Sử dụng id2item để lấy tên sản phẩm từ ID
    product_name = id2item_real.get(recommended_product, "Unknown")
    print("name:", product_name)

"""
root_path = os.path.dirname(os.path.abspath(__file__)) + "\\"
X = pd.read_csv(root_path+'X_matrix.csv',index_col=0)
correlation_matrix = np.load(root_path+'correlation_matrix.npy')
print("A")
print("root_path: ", root_path)
# Đọc tham số truyền từ controller để lấy sp trong đơn hàng gần nhất của user
input_product_str = sys.argv[1]

print(sys.argv[1])
# Loại bỏ dấu ngoặc vuông và dấu cách trong chuỗi
cleaned_str = input_product_str.replace("[", "").replace("]", "").replace(" ", "")

# Tách chuỗi thành danh sách các mã sản phẩm
product_list = cleaned_str.split(",")

# Duyệt qua danh sách sản phẩm đầu vào và đề xuất cho từng sản phẩm
for product_name in product_list:
    product_names = lists(X.index)
    if product_name in product_names:
        product_ID = product_names.index(product_name)
        correlation_product_ID = correlation_matrix[product_ID]

        Recommend = list(X.index[correlation_product_ID > 0.7])

        # Loại bỏ sản phẩm đã mua
        Recommend.remove(product_name)

        # In danh sách sản phẩm đề xuất cho mỗi sản phẩm đầu vào
        for recommended_product in Recommend:
            print("name:", recommended_product)
"""