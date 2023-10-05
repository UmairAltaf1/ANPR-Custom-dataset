import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
data = tf.keras.utils.image_dataset_from_directory('/content/gdrive/MyDrive/posneg')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
  ax[idx].imshow(img.astype(int))
  ax[idx].title.set_text(batch[1][idx])
scaled = batch[0]/255
data = data.map(lambda x,y: (x/255,y))
scaled_iterator = data.as_numpy_iterator()
train_size =int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre= Precision()
re= Recall()
acc= BinaryAccuracy()
for batch in test.as_numpy_iterator():
  X, y = batch
  yhat = model.predict(X)
  pre.update_state(y, yhat)
  re.update_state(y, yhat)
  acc.update_state(y, yhat)
print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')
import cv2
np.expand_dims(resize,0)
np.expand_dims(resize, 0).shape
yhat=model.predict(np.expand_dims(resize/255,0))
if yhat > 0.5:
  print(f'Predicted class is Postive')
else:
    print(f'Predicted class is negative')
from tensorflow.keras.models import load_model
model.save(os.path.join('/content/gdrive/MyDrive/MLModel','firsttarinedmodel.h5'))
