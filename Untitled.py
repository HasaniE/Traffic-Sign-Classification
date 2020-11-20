#!/usr/bin/env python
# coding: utf-8

# # -Traffic Sign Classification System-

# ## import libraries

# In[58]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random


# In[ ]:





# In[60]:


with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)


# In[61]:


X_train,y_train=train['features'],train['labels']
X_validation,y_validation=valid['features'],valid['labels']
X_test,y_test=test['features'],test['labels']


# In[62]:


X_train.shape


# In[63]:


y_train.shape


# ## image visualization

# In[64]:


i = np.random.randint(1, len(X_train))
plt.imshow(X_train[i])
y_train[i]


# In[ ]:





# In[65]:


Width_grid = 5
Length_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)

fig, axes = plt.subplots(Length_grid, Width_grid, figsize = (10,10))

axes = axes.ravel() 

n_training = len(X_train) 

for i in np.arange(0, Width_grid * Length_grid):
   index=np.random.randint(0, n_training)    
   axes[i].imshow(X_train[index])
   axes[i].set_title(y_train[index],fontsize=15)
   axes[i].axis('off')
   
plt.subplots_adjust(hspace=0.4)


# ## image normalization

# In[66]:


from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


# In[67]:


X_train_gray=np.sum(X_train/3,axis=3,keepdims=True)
X_test_gray=np.sum(X_test/3,axis=3,keepdims=True)
X_validation_gray=np.sum(X_validation/3,axis=3,keepdims=True)


# In[68]:


X_train_gray.shape


# In[69]:


X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128


# In[ ]:





# In[70]:


X_train_gray_norm


# In[71]:


X_test_gray_norm


# In[72]:


X_validation_gray_norm


# In[73]:


i = random.randint(1, len(X_train_gray))
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap = 'gray')


# ## building CNN model

# In[74]:


from tensorflow.keras import datasets, layers, models
CNN=models.Sequential()

CNN.add(layers.Conv2D(6,(5,5), activation ='relu',input_shape=(32,32,1)))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Dropout(0.2))

CNN.add(layers.Conv2D(16,(5,5), activation ='relu'))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())

CNN.add(layers.Dense(120, activation ='relu'))
CNN.add(layers.Dense(84, activation ='relu'))
CNN.add(layers.Dense(43, activation ='softmax'))
CNN.summary()


# ## training CNN model

# In[75]:


CNN.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[76]:


history=CNN.fit(X_train_gray_norm,
               y_train,
               batch_size=500,
               epochs= 50,
               verbose=1,
               validation_data=(X_validation_gray_norm, y_validation))


# ## performance of CNN model

# In[77]:


score = CNN.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy: {}'.format(score[1]))


# In[78]:


history.history.keys()


# In[79]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[80]:


epochs=range(len(accuracy))
plt.plot(epochs,loss,'ro',label='training loss')
plt.plot(epochs,val_loss,'r',label='validation loss')
plt.title('taraining and validation loss')


# In[81]:


epochs=range(len(accuracy))
plt.plot(epochs,accuracy,'ro',label='training accuracy')
plt.plot(epochs,val_accuracy,'r',label='validation accuracy')
plt.title('taraining and validation accuracy')


# In[ ]:





# In[82]:


predicted_classes = CNN.predict_classes(X_test_gray_norm)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)


# In[83]:


L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)    


# In[ ]:





# In[ ]:





# In[ ]:




