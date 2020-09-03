### importing necessary libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import glob
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.vq import vq


### reading required train datasets
path = "/home/subarna/Documents/vr/MT2019514_MT2019523_MT2019079/Part2/CIFAR-10-images-master/train"
filenames = glob.glob(path + "/*")
df_train = []
descriptor = []

for filename in filenames:
    clas_nams = glob.glob(filename + "/*")
    for clas_nam in clas_nams:
        try:
            imageID = filename[filename.rfind("/") + 1:]
            img = cv2.imread(clas_nam)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            extractor = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = extractor.detectAndCompute(img, None)
            descriptor.extend(descriptors)
            df_train.append((imageID,descriptors))
        except TypeError as e:
            print(e)

### reading required test dataset
path = "/home/subarna/Documents/vr/MT2019514_MT2019523_MT2019079/Part2/CIFAR-10-images-master/test"
filenames = glob.glob(path + "/*")
df_test = []
descriptor_test = []
for filename in filenames:
    clas_nams = glob.glob(filename + "/*")
    for clas_nam in clas_nams:
        try:
            imageID = filename[filename.rfind("/") + 1:]
            img = cv2.imread(clas_nam)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            extractor = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = extractor.detectAndCompute(img, None)
            descriptor_test.extend(descriptors)
            df_test.append((imageID,descriptors))
        except TypeError as e:
            print(e)



### extracting our clusters and there centroids
kmeans = KMeans(n_clusters = 40, random_state = 10)
kmeans.fit(des_train)
cluster = kmeans.cluster_centers_



### Labelling class names
df_tr = pd.DataFrame(df_train)
df_te = pd.DataFrame(df_test)
l = {'dog' : 0, 'automobile' : 1, 'bird' : 2, 'airplane' : 3, 'ship' : 4, 'truck' : 5, 'frog' : 6, 'horse' : 7, 'deer' : 8, 'cat' : 9}
df_tr[0] = [l[item] for item in df_tr[0]]
df_te[0] = [l[item] for item in df_te[0]]



### getting train and test matrix using numpy
x_test = df_train[0].to_numpy()
y_test = df_test[0].to_numpy()



### creating histogram using BoVW
train_features = np.zeros((len(df_train), 40), "float32")
for i in range(0,len(df_train)):
    words, distance = vq(df_train[i][1],kmeans.cluster_centers_)
    for w in words:
        train_features[i][w] += 1

test_features = np.zeros((len(df_test), 40), "float32")
for i in range(0,len(df_test)):
    words, distance = vq(df_test[i][1],kmeans.cluster_centers_)
    for w in words:
        test_features[i][w] += 1



### preproceesing train and test dataset 
from sklearn.preprocessing import StandardScaler
#Perform Tf-Idf vectorization
nbr_occurences = np.sum( (train_features > 0) * 1, axis = 0)
# Calculating the number of occurrences
idf = np.array(np.log((1.0*len(train_features)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Giving weight to one that occurs more frequently
# Scaling the features
stdSlr = StandardScaler().fit(train_features)
im_features_train = stdSlr.transform(train_features)

nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(test_features)+1) / (1.0*nbr_occurences + 1)), 'float32')
stdSlr = StandardScaler().fit(test_features)
im_features_test = stdSlr.transform(test_features)


### finally getting our train data and test data for our KNN model
x_train = im_features_train
y_train = im_features_test


### finally training our KNN model
model = KNeighborsClassifier(n_neighbors=9,n_jobs=-1)
model.fit(x_train, x_test)
acc = model.score(y_train, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))


### VLAD extension of BoVW
vlad_features = []
for i in range(len(df_train)):
    vlad_vector = np.zeros((40,128), "float32")
    words, distance = vq(df_train[i][1],kmeans.cluster_centers_)
    j=0
    for w in words:
        vlad_vector[w] = np.add(vlad_vector[w], np.subtract(df_train[i][1][j],kmeans.cluster_centers_[w]))
        j = j+1
    norm = np.linalg.norm(vlad_vector)
    vlad_vector= np.divide(vlad_vector,norm)
    vlad_vector= vlad_vector.flatten()
    vlad_features.append(vlad_vector)

vlad_features_test=[]
for i in range(len(df_test)):
    vlad_vector = np.zeros((40,128), "float32")
    words, distance = vq(df_test[i][1],kmeans.cluster_centers_)
    j=0
    for w in words:
        vlad_vector[w] = np.add(vlad_vector[w], np.subtract(df_test[i][1][j],kmeans.cluster_centers_[w]))
        j = j+1
    norm = np.linalg.norm(vlad_vector)
    vlad_vector= np.divide(vlad_vector,norm)
    vlad_vector= vlad_vector.flatten()
    vlad_features_test.append(vlad_vector)



### model training
model = KNeighborsClassifier(n_neighbors=9,n_jobs=-1)
model.fit(vlad_features, x_test)
acc = model.score(vlad_features_test, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))
