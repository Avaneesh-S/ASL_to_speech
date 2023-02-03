import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#getting labels that is names of the people (from folder name)
names=[]
for foldername in os.listdir(r'D:\programming\SE_project_files\asl_images'):
    names.append(foldername)

DIR=r'D:\programming\SE_project_files\asl_images'

features=np.zeros((13000,90,90,3))
features.astype(np.uint8)

labels=[]

#creating training dataset :
def create_train():
    index=0

    for person in names:
        path=os.path.join(DIR,person)
        label=names.index(person)
        count = 0
        for i in range(0,len(os.listdir(path))):
            img=os.listdir(path)[i]
            count+=1
            #getting every image path
            img_path=os.path.join(path,img)
            #getting image matrix
            img_array=cv.imread(img_path)

            img_array=cv.resize(img_array,(90,90),interpolation=cv.INTER_CUBIC)
            features[index] = img_array
            labels.append(label)

            index+=1

'''
DIR_test=r'D:\programming\hand\asl_alphabet_test'

features_test=np.zeros((28,60,60,3))
features_test.astype(np.uint8)

labels_test=[]

#creating testing dataset:
def create_test():
    index=0
    for sign in os.listdir(DIR_test):
        temp = ""
        for i in sign:
            if i == '.':
                break
            else:
                temp += i
        label=names.index(temp)
        #getting every image path
        img_path=os.path.join(DIR_test,sign)
        #getting image matrix
        img_array=cv.imread(img_path)
        img_array = cv.resize(img_array, (60,60), interpolation=cv.INTER_CUBIC)
        features[index] = img_array
        labels_test.append(label)

        index += 1'''

create_train()
labels=np.array(labels)
labels=labels.astype(np.uint8)
X_train,rest_X,Y_train,rest_Y=train_test_split(features,labels,test_size=0.3)
#splitting the rest:
X_val,X_test,Y_val,Y_test=train_test_split(rest_X,rest_Y,test_size=0.4)

#creating CNN:
model=keras.Sequential([
    keras.layers.Conv2D(40,(4,4),input_shape=(90,90,3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(65,(3,3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(65,(3,3)),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(120,activation="relu"),
    keras.layers.Dense(len(names),activation="softmax")

])

early_stop=EarlyStopping(monitor='val_loss',patience=3)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=6,callbacks=[early_stop],validation_data=(X_val,Y_val))

'''
NOTE:
callbacks are objects which perform some action
'''

test_loss,test_accuracy=model.evaluate(X_test,Y_test)
test_accuracy=test_accuracy*100

print("accuracy = ",test_accuracy,'%')

if(test_accuracy>=90):
    model.save('hand_recognise_model_own90_'+str(int(test_accuracy))+'_.h5')
    print("model saved")






