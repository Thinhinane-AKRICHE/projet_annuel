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


# In[4]:


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


# In[5]:


# Charger la bibliothèque Rust
lib = ctypes.CDLL("target/release/mymodel.dll")

lib.create_mlp_classifier.argtypes = [
    ctypes.c_size_t,  # n_inputs
    ctypes.c_size_t,  # n_hidden
    ctypes.c_size_t,  # n_classes
    ctypes.c_double,  # learning_rate
    ctypes.c_size_t,  # epochs
    ctypes.c_uint32,  # activation_id
    ctypes.c_size_t,  # batch_size
    ctypes.c_double   # lambda
]


lib.create_mlp_classifier.restype = ctypes.c_void_p
lib.train_mlp_classifier.argtypes = [
    ctypes.c_void_p,                   # model_ptr
    ctypes.POINTER(ctypes.c_double),  # x_ptr
    ctypes.POINTER(ctypes.c_uint32),  # y_ptr
    ctypes.c_size_t,                  # n_samples
    ctypes.c_size_t                   # n_features
]
lib.train_mlp_classifier.restype = None

lib.predict_mlp_classifier.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
lib.predict_mlp_classifier.restype = ctypes.c_uint32


# In[6]:


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


# In[7]:


# === Prétraitement pour MLP classifier (X: float64, y: uint32) ===
def preprocess_for_mlp(images, labels):
    X = np.stack([transform(img).numpy().flatten() for img in images])
    X = X.astype(np.float64)

    le = LabelEncoder()
    y_int = le.fit_transform(labels).astype(np.uint32)

    return X, y_int, le


# In[8]:


# Chargement séparé
images_train, labels_train = load_images_from_folder("dataset/train")
images_test, labels_test = load_images_from_folder("dataset/test")

# Prétraitement
X_train, Y_train, label_encoder = preprocess_for_rust(images_train, labels_train)
X_test, Y_test, _ = preprocess_for_rust(images_test, labels_test)

n_train, n_features = X_train.shape
n_outputs = Y_train.shape[1]



# In[16]:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



print("X_train dtype :", X_train.dtype)
print("X_train shape :", X_train.shape)
print("X_train min :", np.min(X_train))
print("X_train max :", np.max(X_train))
print("X_train[0][:10] :", X_train[0][:10])



# === Créer le modèle MLP ===
hidden_units = 64
learning_rate = 0.001
epochs = 200
batch_size = 32
lambda_ = 0.001

model_ptr = lib.create_mlp_classifier(
    n_features,       # n_inputs
    hidden_units,     # n_hidden
    3,                # n_classes
    learning_rate,    # learning_rate
    epochs,           # epochs
    0,                # activation_id (0 = ReLU)
    batch_size,       # batch_size
    lambda_           # lambda
)


# === Entraînement ===
x_train_ptr = X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

y_indices = np.argmax(Y_train, axis=1).astype(np.uint32)
y_train_ptr = y_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

lib.train_mlp_classifier(model_ptr, x_train_ptr, y_train_ptr, n_train, n_features)



# In[17]:


# --- Fonction de prédiction ---
def predict_image(model_ptr, image):
    x = transform(image).numpy().flatten().astype(np.float64)
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pred = int(lib.predict_mlp_classifier(model_ptr, x_ptr, x.shape[0]))
    return label_encoder.inverse_transform([pred])[0]


# In[18]:


# --- Évaluation sur le test set ---
y_pred = [predict_image(model_ptr, img) for img in images_test]
# --- Résultat ---
print("✅ Accuracy sur données de test :", accuracy_score(labels_test, y_pred))


# In[19]:


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


# In[20]:


afficher_toutes_les_predictions(images_test, labels_test, model_ptr)


# In[15]:


from collections import Counter
print("Train set :", Counter(labels_train))
print("Test set :", Counter(labels_test))


# In[22]:


y_preds = [lib.predict_mlp_classifier(model_ptr, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n_features)
           for x in X_test]

print("Prédictions:", y_preds)
print("Labels vrais:", Y_test.tolist())

