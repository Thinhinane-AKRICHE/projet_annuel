#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from sklearn.preprocessing import StandardScaler
# In[2]:


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


# In[3]:


# Charger la bibliothèque Rust
lib = ctypes.CDLL("target/release/mymodel.dll")

# === Définir les signatures ===
lib.create_softmax_model.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t, ctypes.c_double
]
lib.create_softmax_model.restype = ctypes.c_void_p

lib.train_softmax_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_size_t,
    ctypes.c_size_t
]
lib.train_softmax_model.restype = None

lib.predict_softmax_model.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_softmax_model.restype = ctypes.c_size_t
lib.predict_rbfn_model.restype = ctypes.c_double


# In[4]:


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


# In[5]:


# Chargement séparé
images_train, labels_train = load_images_from_folder("dataset/train")
images_test, labels_test = load_images_from_folder("dataset/test")

# Prétraitement
X_train, Y_train, label_encoder = preprocess_for_rust(images_train, labels_train)
X_test, Y_test, _ = preprocess_for_rust(images_test, labels_test)

n_train, n_features = X_train.shape
n_outputs = Y_train.shape[1]


# In[15]:

#normalisation de deonnées
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train dtype :", X_train.dtype)
print("X_train shape :", X_train.shape)
print("X_train min :", np.min(X_train))
print("X_train max :", np.max(X_train))
print("X_train[0][:10] :", X_train[0][:10])


# --- Créer le modèle RBFN ---
model_ptr = lib.create_softmax_model(n_features, 3, 0.1, 1000, 0.0)

# --- Entraîner le modèle ---
x_train_ptr = X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
y_indices = np.argmax(Y_train, axis=1).astype(np.uint64)
y_train_ptr = y_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


lib.train_softmax_model(model_ptr, x_train_ptr, y_train_ptr, n_train, n_features)


# In[16]:


# --- Fonction de prédiction ---
def predict_image(model_ptr, image):
    x = transform(image).numpy().flatten().astype(np.float64)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pred = int(lib.predict_softmax_model(model_ptr, x_ptr, x.shape[0]))
    return label_encoder.inverse_transform([pred])[0]


# In[17]:


# --- Évaluation sur le test set ---
y_pred = [predict_image(model_ptr, img) for img in images_test]
# --- Résultat ---
print("✅ Accuracy sur données de test :", accuracy_score(labels_test, y_pred))


# In[18]:


def afficher_toutes_les_predictions(images, labels, model_ptr):
    n = len(images)  # nombre total d’images à afficher
    cols = 10        # nombre d’images par ligne
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

        pred = predict_image(model_ptr, images[i])
        true = labels[i]

        color = "green" if pred == true else "red"
        plt.title(f"P: {pred}\nR: {true}", fontsize=8, color=color)

    plt.tight_layout()
    plt.show()


# In[19]:


afficher_toutes_les_predictions(images_test, labels_test, model_ptr)


# In[14]:


y_preds = [lib.predict_softmax_model(model_ptr, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_features)
           for x in X_test]

print("Prédictions:", y_preds)
print("Labels vrais:", Y_test.tolist())

