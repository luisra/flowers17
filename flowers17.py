#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:07:30 2017

@author: LuisRa
"""

# Module check/download - those not in Python 3.5.3 #

try:
    from pip import main as pip_main
except:
    from pip._internal import main as pip_main

def import_or_install(package):
    try:
        __import__(package)
        print(i+": "+"ok")
    
    except ImportError:
        print("\n"+i+": "+"installing...")
        pip_main(['install', package])
        print(i+": "+"ok")

modules = ['numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas',
           'tensorflow', 'scikit-learn', 'opencv-python']

for i in modules:
    import_or_install(i)

#

# Relevant libraries

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.python.platform import gfile
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
import tarfile
import urllib
import cv2
import os
import re

#

# Data retrieval (Desktop wil be used as workspace.)

def get_data():
    
    url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
    print ("\ndownloading flower images...")    
    filename, headers = urllib.request.urlretrieve(
        url,filename=os.path.expanduser("~/Desktop/17flowers.tgz"))    
    print ("download complete!")

    os.chdir(os.path.expanduser("~/Desktop/"))
    print ("extracting flower images...")
    tar = tarfile.open(os.path.expanduser("~/Desktop/17flowers.tgz"), "r:gz")
    tar.extractall()
    tar.close()
    print ("extract complete!")
    
    print ("downloading tensorflow component...")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/imagenet/classify_image.py",
                           filename=os.path.expanduser("~/Desktop/classify_image.py"))
    print ("download complete!")
    
    os.chdir(os.path.expanduser("~/Desktop/"))
    print ("generating graph...")
    os.system("python classify_image.py --model_dir ~/Desktop/graph/")
    print ("graph complete!\n")

get_data()

#

# Classes

classes = ['Daffodil','Snowdrop', 'Lily Valley', 'Bluebell',
           'Crocus', 'Iris', 'Tigerlily', 'Tulip',
           'Fritillary', 'Sunflower', 'Daisy', 'Colts Foot',
            'Dandelalion', 'Cowslip', 'Buttercup', 'Windflower',
            'Pansy']

y = np.repeat(classes, 80)

#

# Image files

images = []
loc = os.path.expanduser("~/Desktop/jpg")

for filename in sorted(os.listdir(loc)):
    
    img = cv2.imread(os.path.join(loc,filename))    
    
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

images = np.asarray(images)

#

# Tensor prep

path = os.path.expanduser("~/Desktop/jpg")
files = sorted(os.listdir(path))
i = 1

for file in files:    # Rename image files
    if re.search("jpg", file):        
        os.rename(os.path.join(path, file), os.path.join(path, y[i-1]+'_'+str(i)+'.jpg'))
        i = i+1     
#

# Tensorflow

model_dir = os.path.expanduser("~/Desktop/graph/") 
os.chdir(os.path.expanduser("~/Desktop/"))
images_dir = "jpg/"

list_images = [images_dir+f for f in os.listdir(images_dir) if re.search("jpg", f)] # List of new image filenames


tmp = []

for i in range(len(list_images)): # Retrieve numbers from filenames
    
    t = re.findall(r"\d+", list_images[i])
    tmp.append(int(t[0]))

list_images = [x for (y,x) in sorted(zip(tmp,list_images))] # Order image files by number


def create_graph(): # Create instance of the trained model
    
    with gfile.FastGFile(os.path.join(model_dir, "classify_image_graph_def.pb"), "rb") as f:
        
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
    
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []
    
    create_graph() # Initiate instance of the trained model

    with tf.Session() as sess: # Retrieve penultimate layer
        penultimate_tensor = sess.graph.get_tensor_by_name('pool_3:0')

    for i in range(len(list_images)): # Feed image into layer and retrieve features and label
        
        if (i%100 == 0):
            print("Processing %s..." % (list_images[i]))
            
        preds = sess.run(penultimate_tensor,
                           {'DecodeJpeg:0': images[i]})
        features[i,:] = np.squeeze(preds)
        labels.append(re.split("_\d+",list_images[i].split("/")[1])[0])

    return features, labels

features, labels = extract_features(list_images)

#

#

model = LinearSVC(C=1, loss='squared_hinge', penalty='l2',multi_class='ovr')

#

# Linear SVC performance/results 

Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels,
                                                random_state = 7,
                                                test_size = 0.3
                                                )

model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)

print("\nLinear SVC Accuracy (Ten-Fold CV):", cross_val_score(model, features, labels, cv=10).mean(), "\n")

print("Linear SVC Accuracy (Holdout Set):", accuracy_score(ytest, ypred), "\n")

print("Linear SVC Classification Report:", "\n")

print(classification_report(ytest, model.predict(Xtest), 
                            target_names = classes))

plt.figure(figsize=(8, 8))
mat = confusion_matrix(ytest, ypred)
ax = sns.heatmap(mat.T, square = True, annot = True, fmt='d', cbar=False,
            xticklabels = classes, yticklabels= classes)
plt.xlabel('true label')
plt.ylabel('pred label')
plt.title('Linear SVC Heatmap')
plt.show();

#
