import tensorflow as tf
import os 
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import precision_score, recall_score, accuracy_score






data_dir = 'images'
image_extensions = ['jpeg', 'jpg', 'png', 'bmp']


# print(os.listdir(os.path.join(data_dir, '4')))


# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         img = cv2.imread(image_path)
#         print(img)





# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

data = tf.keras.utils.image_dataset_from_directory('images', batch_size = 3, image_size=(256, 256))



    
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

# print(batch[1])

# fig, ax = plt.subplots(ncols = 3, figsize = (10,10))
# for idx, img in enumerate(batch[0][:3]):
#     # ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])

scaled = batch[0] / 255.0

# print(scaled.min())

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

training_size = int(len(data)*.7)
validation_size = int(len(data)*.2)+1
testing_size = int(len(data)*.1)+1


train = data.take(training_size)
validation = data.skip(training_size).take(validation_size)
test = data.skip(training_size+validation_size).take(testing_size)


# print(len(train))
# print(len(validation))
# print(len(test))



model = Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(150, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))



model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=15, validation_data=validation, callbacks=[tensorboard_callback])


# fig = plt.figure()
# plt.plot(hist.history['loss'], color = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], color = 'blue', label = 'val_loss')
# fig.suptitle('Loss', fontsize = 20)
# plt.legend(loc = 'upper left')
# plt.show()



# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color = 'red', label = 'accuracy')
# plt.plot(hist.history['val_accuracy'], color = 'blue', label = 'val_accuracy')
# fig.suptitle('Accuracy', fontsize = 20)
# plt.legend(loc = 'upper left')
# plt.show()




y_true = []
y_pred = []

for X, y in test:
    yhat = model.predict(X)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(yhat, axis=1))

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute metrics
pre = precision_score(y_true, y_pred, average='macro')
re = recall_score(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)




# resize = tf.image.resize(img, (256, 256))
# plt.imshow(resize.numpy().astype(int))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

# np.expand_dims(resize, 0)
# yhat = model.predict(np.expand_dims(resize/255, 0))
# print(yhat * 100)


img = cv2.imread('test_Image.jpeg')
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
input = np.expand_dims(resize / 255.0, 0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

yhat = model.predict(input)
# print(yhat * 100)

# plt.show()

#Saving the model
from keras.models import load_model
model.save(os.path.join('Completed_Models', 'image_Classification.keras'))


