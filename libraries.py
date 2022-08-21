import pandas as pd
import numpy as np
from PIL import Image
import urllib.request
import os
import cv2
from retinaface import RetinaFace
from deepface import DeepFace
import requests # request img from web
import shutil # save img locally
import random



def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    face_array = []
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    #plt.show()
    face = RetinaFace.extract_faces(img_path = filename, align = True)
    if len(face)>0:
        face = face[0]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        #plt.imshow(image)
        face_array = np.asarray(image)
    else:
        print(f'{filename} Invalid')
    return face_array

def get_embedding(model, filename, distance = 'cosine'):
    # scale pixel values
    face = extract_face(filename)
    embd = []
    if face!=[]:
        face = face.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face-mean)/std
        # transfer face into one sample (3 dimension to 4 dimension)
        face = np.expand_dims(face, axis=0)
        # Get embedding
        embd = model.predict(face)[0,:]
        if distance == 'eu':
            embd = l2_normalize(embd)
    return embd

def load_embeddings(path, model, dist = 'cosine'):
    emdngs = list()
    labels = []
    # enumerate files
    for filename in os.listdir(path):
        fname = path +"/"+ filename
        print(fname)
        emdng = get_embedding(model, fname, dist)
        if emdng != []:
            emdngs.append(emdng)
            labels.append(filename.split(".")[0])
        else:
            print(f'{fname} Invalid')
    return np.asarray(emdngs),labels
