import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from time import time
import scipy.sparse as sp

import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import random
# Thay đổi các thông số kết nối theo cơ sở dữ liệu SQL Server của bạn
"""server = 'HP'
database = 'shopThoiTrang'
username = 'sa'
password = '123456'

# Tạo một URI kết nối sử dụng SQLAlchemy
connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server"

# Tạo một kết nối đến cơ sở dữ liệu
engine = create_engine(connection_string)

table_name = 'DANHGIA'

# Chỉ định cột quan tâm
columns_to_export = ['MAND', 'MASP', 'SOSAO']

# Sử dụng pandas để đọc dữ liệu từ bảng
sql_query = f'SELECT {", ".join(columns_to_export)} FROM {table_name}'
df = pd.read_sql(sql_query, engine)

ratings_utility_matrix = df.pivot_table(values='SOSAO', index='MAND', columns='MASP', fill_value=0)
# print(ratings_utility_matrix.head())

X = ratings_utility_matrix.T

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

# correlation_matrix = np.corrcoef(decomposed_matrix)


# Tạo ma trận tương quan
correlation_matrix = np.corrcoef(decomposed_matrix)

# Lưu ma trận tương quan vào tệp
np.save('correlation_matrix.npy', correlation_matrix)

X.to_csv('X_matrix.csv')
print("Da train va luu du lieu")
"""
import os
import pandas as pd
import numpy as np
import random
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

# Constants for paths and database configurations
DATABASE_CONFIG = {
    "server": "HP",
    "database": "shopThoiTrang",
    "username": "sa",
    "password": "123456",
}
FILE_PATHS = {
    "correlation_matrix": "correlation_matrix.npy",
    "X_matrix": "X_matrix.csv",
    "ratings_file": "/content/drive/MyDrive/Colab Notebooks/dataset/ml-1m/ratings.dat",
    "movies_file": "/content/drive/MyDrive/Colab Notebooks/dataset/ml-1m/movies.dat",
    "users_file": "/content/drive/MyDrive/Colab Notebooks/dataset/ml-1m/users.dat",
    "train_rating": "/content/train.rating",
    "test_rating": "/content/test.rating",
    "test_negative": "/content/test.negative",
    "weights": "ratings.csv"
}

latent_dims = 4
layers = [16,8,4]
epochs = 50
batch_size = 256
learning_rate=1e-3

SEED = 42

num_negatives = 4 #ratio positive : negative - 1:4
# SQLAlchemy connection string
connection_string = f"mssql+pyodbc://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['server']}/{DATABASE_CONFIG['database']}?driver=SQL+Server"

# Database table and columns to retrieve
TABLE_NAME = 'DANHGIA'
COLUMNS_TO_EXPORT = ['MAND', 'MASP', 'SOSAO']
COLUMNS_USER_TO_EXPORT = ['MAND', 'GIOITINH', 'NGAYSINH']
COLUMNS_ITEM_TO_EXPORT = ['MASP', 'DONGIA', 'TRANGTHAI', 'MAKIEU']
# Load data from SQL
def load_data_from_sql():
    engine = create_engine(connection_string)
    sql_query = f'SELECT {", ".join(COLUMNS_TO_EXPORT)} FROM {TABLE_NAME}'
    return pd.read_sql(sql_query, engine)

def load_user_from_sql():
    engine = create_engine(connection_string)
    sql_query = f'SELECT {", ".join(COLUMNS_USER_TO_EXPORT)} FROM NGUOIDUNG'
    return pd.read_sql(sql_query, engine)

def load_item_from_sql():
    engine = create_engine(connection_string)
    sql_query = f'SELECT {", ".join(COLUMNS_ITEM_TO_EXPORT)} FROM SANPHAM'
    return pd.read_sql(sql_query, engine)

    
def write_data_to_csv(df, output_name):
	df.to_csv(output_name, sep = '\t', index = False, header = False)
	
def get_rating_data():
    # Load data using the load_data_from_sql function
    df = load_data_from_sql()
    
    # Thực hiện xử lý bổ sung nếu cần, ví dụ như kiểm tra null hoặc chuẩn hóa dữ liệu
    if df.isnull().values.any():
        df = df.dropna()  # Xóa các hàng có giá trị null
    
    # Giả sử rating đã có trong dữ liệu, chỉ cần trả lại DataFrame kết quả
    return df


def map_item_to_id(data):
    # Tạo danh sách các masp duy nhất từ dữ liệu
    unique_items = data['masp'].unique()

    # Tạo từ điển ánh xạ masp ban đầu thành các ID liên tiếp
    item2id = {item: idx for idx, item in enumerate(unique_items)}

    # Tạo từ điển ngược lại để khôi phục masp ban đầu
    id2item = {idx: item for item, idx in item2id.items()}

    # Thay thế masp trong data bằng new_id
    data['masp'] = data['masp'].map(item2id)

    return data, item2id, id2item

# Hàm để khôi phục lại masp ban đầu từ new_id
def recover_original_item(mapped_data, id2item):
    # Thay thế new_id bằng masp ban đầu
    mapped_data['masp'] = mapped_data['masp'].map(id2item)
    return mapped_data

# Hàm 1: Tách dữ liệu thành tập huấn luyện và tập kiểm tra
def split_data(ratings_file, test_ratio=0.2):
    # Đọc file ratings.dat
    # ratings = pd.read_csv(ratings_file, sep="::", engine="python", names=["mand", "masp", "rating", "timestamp"])

    # Chia dữ liệu: giữ lại một rating cho mỗi user vào tập kiểm tra, phần còn lại vào tập huấn luyện
    test_ratings = ratings.groupby("mand").sample(n=1, random_state=42)  # Mỗi user có một dòng vào tập kiểm tra
    train_ratings = ratings[~ratings.index.isin(test_ratings.index)]  # Phần còn lại là tập huấn luyện

    return train_ratings, test_ratings

# Hàm 2: Tạo file train.rating
def generate_train_rating(train_ratings, output_file="/content/train.rating"):
    # Lưu file train.rating với định dạng `userID\t itemID\t rating\t timestamp`
    train_ratings.to_csv(output_file, sep="\t", header=False, index=False)

# Hàm 3: Tạo file test.rating và test.negative
def generate_test_rating_and_negative(test_ratings, train_ratings, num_negatives=99, test_rating_file="/content/test.rating", test_negative_file="/content/test.negative"):
    # Lưu tập kiểm tra (positive instances)
    test_ratings.to_csv(test_rating_file, sep="\t", header=False, index=False)

    # Lấy danh sách các item có trong tập huấn luyện để xác định negative samples
    num_items = train_ratings['masp'].max()

    with open(test_negative_file, 'w') as f_neg:
        for _, row in test_ratings.iterrows():
            mand = row['mand']
            masp = row['masp']
    
            # Lấy các items chưa được đánh giá bởi user để tạo negative samples
            positive_items = set(train_ratings[train_ratings['mand'] == mand]['masp'].values)
            negative_items = set(range(1, num_items + 1)) - positive_items  # Tất cả items trừ items đã đánh giá
    
            # Chuyển negative_items thành danh sách (list)
            negative_items_list = list(negative_items)
    
            # Chọn ngẫu nhiên num_negatives samples từ các negative_items
            sampled_negatives = random.sample(negative_items_list, min(len(negative_items_list), num_negatives))
    
            # Lưu vào file theo định dạng: (userID, itemID) \t negativeItemID1 \t negativeItemID2 ...
            f_neg.write(f"({mand},{masp})\t" + "\t".join(map(str, sampled_negatives)) + "\n")

def get_model(num_users, num_items):
  user_input = Input(shape=(1,), dtype='int32', name='user_input')
  item_input = Input(shape=(1,), dtype='int32', name='item_input')
  # item_feature_input = Input(shape=(4,), dtype='float32', name='item_feature_input')  # Item features vector


  GMF_Embedding_User = Embedding(input_dim = num_users + 1, output_dim = latent_dims, embeddings_initializer= initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    embeddings_regularizer=regularizers.l2(0),
                                    name='gmf_user_embedding')(user_input)
  GMF_Embedding_Item = Embedding(input_dim=num_items + 1, output_dim=latent_dims,
                                embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                embeddings_regularizer=regularizers.l2(0),
                                name='gmf_item_embedding')(item_input)

  gmf_user_latent = Flatten()(GMF_Embedding_User)
  gmf_item_latent = Flatten()(GMF_Embedding_Item)
  gmf_vector = Multiply()([gmf_user_latent, gmf_item_latent])
  gmf_vector = Dropout(0.12)(gmf_vector)
  # gmf_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='gmf_prediction')(gmf_vector)
  # gmf_model = Model(inputs=[gmf_user_input, gmf_item_input], outputs=prediction)
  #
  # gmf_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

  num_layer = len(layers) #Number of layers in the MLP
  # Input variables
  # user_mlp_input = Input(shape=(1,), dtype='int32', name = 'user_input')
  # item_mlp_input = Input(shape=(1,), dtype='int32', name = 'item_input')

  MLP_Embedding_User = Embedding(input_dim = num_users + 1, output_dim = layers[0]//2, name = 'user_embedding',
                                embeddings_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01), embeddings_regularizer = l2(layers[0]), input_length=1)(user_input)
  MLP_Embedding_Item = Embedding(input_dim = num_items + 1, output_dim = layers[0]//2, name = 'item_embedding',
                                embeddings_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01), embeddings_regularizer = l2(layers[0]), input_length=1)(item_input)

  mlp_user_latent = Flatten()(MLP_Embedding_User)
  mlp_item_latent = Flatten()(MLP_Embedding_Item)

  mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
  for idx in range(1, num_layer):
    layer = Dense(layers[idx-1], kernel_initializer='lecun_uniform', activation='sigmoid', name = 'layer%d' %idx)(mlp_vector)

    mlp_vector = mlp_vector = Dropout(0.12)(layer)



  # mlp_vector = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(mlp_vector)

  # model = Model(inputs=[user_mlp_input, item_mlp_input],
  #               outputs=prediction)

  # model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

  # model.summary()

  neumf_vector = Concatenate()([gmf_vector, mlp_vector])
  prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(neumf_vector)

  model  = Model(inputs=[user_input, item_input],
                outputs=prediction)
  return model

def evaluate_model(model, testRatings, testNegatives, K, num_thread):


    hits, ndcgs = [],[]

    for idx in range(len(testRatings)):
        (hr,ndcg) = eval_one_rating(model, idx, testRatings, testNegatives, K)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def eval_one_rating(model, idx, testRatings, testNegatives, K):
    rating = testRatings[idx]
    items = testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getPrecision(ranklist, gtItem, K):
    return 1 / K if gtItem in ranklist else 0

def getRecall(ranklist, gtItem):
  return 1 if gtItem in ranklist else 0

def load_rating_file_as_list(filename):
  ratingList = []
  with open(filename, "r") as f:
      line = f.readline()
      while line != None and line != "":
          arr = line.split("\t")
          user, item = int(arr[0]), int(arr[1])
          ratingList.append([user, item])
          line = f.readline()
  return ratingList

def load_negative_file(filename):
  negativeList = []
  with open(filename, "r") as f:
      line = f.readline()
      while line != None and line != "":
          arr = line.split("\t")
          negatives = []
          for x in arr[1: ]:
              negatives.append(int(x))
          negativeList.append(negatives)
          line = f.readline()
  return negativeList

def load_rating_file_as_matrix(filename):
  '''
  Read .rating file and Return dok matrix.
  The first line of .rating file is: num_users\t num_items
  '''
  # Get number of users and items
  num_users, num_items = 0, 0
  with open(filename, "r") as f:
      line = f.readline()
      while line != None and line != "":
          arr = line.split("\t")
          u, i = int(arr[0]), int(arr[1])
          num_users = max(num_users, u)
          num_items = max(num_items, i)
          line = f.readline()
  # Construct matrix
  mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
  with open(filename, "r") as f:
      line = f.readline()
      while line != None and line != "":
          arr = line.split("\t")
          user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
          if (rating > 0):
              mat[user, item] = 1.0
          line = f.readline()
  return mat

# train_data = load_rating_file_as_matrix("/content/drive/MyDrive/Colab Notebooks/dataset/ml-1m/ml-1m.train.rating")
# test_data = load_rating_file_as_list("/content/drive/MyDrive/Colab Notebooks/dataset/ml-1m/ml-1m.test.rating")
# test_negative = load_negative_file("/content/drive/MyDrive/Colab Notebooks/dataset/ml-1m/ml-1m.test.negative")
# def get_train_instances(train_data, num_users, num_items, num_negatives=4):
#     user_input, item_input, labels = [], [], []
#     mk = 0
#     # Iterate over each row in the DataFrame
#     for _, row in train_data.iterrows():
#         u, i = int(row['mand']), int(row['masp'])

#         # Positive instance
#         user_input.append(u)
#         item_input.append(i)
#         labels.append(1)

#         # Negative instances
#         for _ in range(num_negatives):
#             j = np.random.randint(num_items)  # Randomly select a negative item
#             while (u, j) in zip(train_data['mand'], train_data['masp']):
#                 j = np.random.randint(num_items)  # Keep picking until a true negative is found
#             if (mk == 100):
#               print("found: ", 100 / len(train_data))
#             user_input.append(u)
#             item_input.append(j)
#             labels.append(0)
#             mk = mk + 1

#     return user_input, item_input, labels

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

DATABASE_URI = 'mssql+pyodbc://sa:2612@localhost:1433/shopThoiTrang?driver=ODBC+Driver+17+for+SQL+Server'

# Tạo engine để kết nối với cơ sở dữ liệu
engine = create_engine(DATABASE_URI)

# Câu lệnh SQL để lấy dữ liệu từ bảng
query = "SELECT mand, masp, sosao FROM DANHGIA"
query_user = "SELECT mand, gioitinh, ngaysinh FROM NGUOIDUNG"
query_item = "SELECT masp, makieu FROM SANPHAM WHERE trangthai = 1"
# Đọc dữ liệu từ SQL vào DataFrame
df = pd.read_sql(query, engine)
user_df = pd.read_sql(query_user, engine)
item_df = pd.read_sql(query_item, engine)
# Đường dẫn tới file output
output_file = 'ratings.csv'
output_file_user = 'users.csv'
output_file_item = 'items.csv'
df.to_csv(output_file, sep='\t', index=False, header=False)
user_df.to_csv(output_file_user, sep='\t', index=False, header=False)
item_df.to_csv(output_file_item, sep='\t', index=False, header=False)


"""
# Generate matrix and correlation
def generate_matrix_and_save(df):
    ratings_matrix = df.pivot_table(values='SOSAO', index='MAND', columns='MASP', fill_value=0)
    X = ratings_matrix.T
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    np.save(FILE_PATHS["correlation_matrix"], correlation_matrix)
    X.to_csv(FILE_PATHS["X_matrix"])
    print("Matrix and correlation saved successfully.")

# Split data for train and test
def split_data(data, test_ratio=0.2):
    test_ratings = data.groupby(CONSTS["mand"]).sample(n=1, random_state=42)
    train_ratings = data[~data.index.isin(test_ratings.index)]
    return train_ratings, test_ratings

# Generate training and test files
def generate_train_and_test_files(train_ratings, test_ratings):
    train_ratings.to_csv(FILE_PATHS["train_rating"], sep="\t", header=False, index=False)
    test_ratings.to_csv(FILE_PATHS["test_rating"], sep="\t", header=False, index=False)

    num_items = train_ratings[CONSTS['masp']].max()
    with open(FILE_PATHS["test_negative"], 'w') as f_neg:
        for _, row in test_ratings.iterrows():
            mand = row[CONSTS['mand']]
            masp = row[CONSTS['masp']]
            positive_items = set(train_ratings[train_ratings[CONSTS['mand']] == mand][CONSTS['masp']].values)
            negative_items = set(range(1, num_items + 1)) - positive_items
            sampled_negatives = random.sample(negative_items, min(len(negative_items), 4))
            f_neg.write(f"({mand},{masp})\t" + "\t".join(map(str, sampled_negatives)) + "\n")
    print("Train and test files generated.")

def map_item_to_id(data):
    # Tạo danh sách các masp duy nhất từ dữ liệu
    unique_items = data[CONSTS['masp']].unique()

    # Tạo từ điển ánh xạ masp ban đầu thành các ID liên tiếp
    item2id = {item: idx for idx, item in enumerate(unique_items)}

    # Tạo từ điển ngược lại để khôi phục masp ban đầu
    id2item = {idx: item for item, idx in item2id.items()}

    # Thay thế masp trong data bằng new_id
    data[CONSTS['masp']] = data[CONSTS['masp']].map(item2id)

    return data, item2id, id2item

# Hàm để khôi phục lại masp ban đầu từ new_id
def recover_original_item(mapped_data, id2item):
    # Thay thế new_id bằng masp ban đầu
    mapped_data[CONSTS['masp']] = mapped_data[CONSTS['masp']].map(id2item)
    return mapped_data

# Main processing
    # Load data from SQL and save matrix and correlation
    data = load_data_from_sql()
    generate_matrix_and_save(data)
	
    # Example of reading and processing ratings
    ratings = pd.read_csv(FILE_PATHS["ratings_file"], sep="::", engine="python", names=["mand", "masp", "rating", "timestamp"])
    train_ratings, test_ratings = split_data(ratings)
    generate_train_and_test_files(train_ratings, test_ratings)
"""
if __name__ == "__main__":
	ratings = pd.read_csv(FILE_PATHS["weights"], sep ="\t", engine = "python", names = ["mand", "masp", "rating"], encoding = "latin-1")
	mapped_ratings = ratings.copy() #tránh thay đổi object gốc
	
	mapped_movies, movie2id, id2movie = map_item_to_id(mapped_ratings)
	mapped_ratings['masp'] = ratings['masp'].map(movie2id)
	num_users = mapped_ratings['mand'].nunique()
	num_items = mapped_ratings['masp'].nunique()
	items = mapped_ratings['masp'].unique()
	# num_users = len(users)
	# num_items = len(movies)
	
	# Chạy các hàm
	# Bước 1: Tách dữ liệu
	train_ratings, test_ratings = split_data(mapped_ratings)
	mapped_train_ratings, item2id_ratings, id2item_ratings = map_item_to_id(train_ratings)
	mapped_test_ratings, item2id_test, id2item_test = map_item_to_id(test_ratings)
	
	# Bước 2: Tạo file train.rating
	generate_train_rating(mapped_train_ratings)
	
	# Bước 3: Tạo file test.rating và test.negative
	generate_test_rating_and_negative(mapped_test_ratings, mapped_train_ratings)
	
	
	test_data = load_rating_file_as_list("test.rating")
	
	test_negative = load_negative_file("test.negative")
	
	matrix = load_rating_file_as_matrix("train.rating")
	
	
	u_inputs, i_inputs, labels = get_train_instances(matrix, 4)
	model = get_model(num_users, num_items)
	model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
	
	model.summary()
	hist = model.fit([np.array(u_inputs), np.array(i_inputs)], #input
			np.array(labels), # labels
			batch_size=256, epochs=10, verbose=1, shuffle=True)
                         
	model.save_weights(FILE_PATHS["weights"])