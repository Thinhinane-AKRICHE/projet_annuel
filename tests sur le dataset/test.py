import ctypes
import numpy as np
import os
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from data_charge import load_images_from_mongodb
#from prepar_data import preprocess_for_rust, transform
import matplotlib.pyplot as plt
import math
from PIL import Image

from torchvision import transforms
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import numpy as np

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def preprocess_for_rust(images, labels):
    X = np.stack([transform(img).numpy().flatten() for img in images])
    X = X.astype(np.float64)

    le = LabelEncoder()
    y_int = le.fit_transform(labels)

    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y_int)
    if y_onehot.shape[1] == 1:  # pour les cas binaire
        y_onehot = np.hstack([1 - y_onehot, y_onehot])

    y_onehot = y_onehot.astype(np.float64)

    return X, y_onehot, le


# Charger la bibliothèque Rust
lib = ctypes.CDLL("target/release/mymodel.dll")

#
lib.create_rbfn_multiclass_model.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t]
lib.create_rbfn_multiclass_model.restype = ctypes.c_void_p

lib.train_rbfn_model_auto.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]

lib.predict_rbfn_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.predict_rbfn_model.restype = ctypes.c_double



def load_images_from_folder(folder_path):
    images, labels = [], []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(label_path, filename)
                try:
                    image = Image.open(image_path).convert("RGB")
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Erreur lors de l'ouverture de {image_path}: {e}")

    return images, labels



# Chargement séparé
images_train, labels_train = load_images_from_folder("dataset/train")
images_test, labels_test = load_images_from_folder("dataset/test")

# Prétraitement
X_train, Y_train, label_encoder = preprocess_for_rust(images_train, labels_train)
X_test, Y_test, _ = preprocess_for_rust(images_test, labels_test)

n_train, n_features = X_train.shape
n_outputs = Y_train.shape[1]


# --- Créer le modèle RBFN ---
model_ptr = lib.create_rbfn_multiclass_model(1.0, 0.01, 100, n_outputs)

# --- Entraîner le modèle ---
x_train_ptr = X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
y_train_ptr = Y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

lib.train_rbfn_model_auto(model_ptr, x_train_ptr, y_train_ptr, n_train, n_features, n_outputs)



# --- Fonction de prédiction ---
def predict_image(model_ptr, image):
    x = transform(image).numpy().flatten().astype(np.float64)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pred = int(lib.predict_rbfn_model(model_ptr, x_ptr, x.shape[0]))
    return label_encoder.inverse_transform([pred])[0]