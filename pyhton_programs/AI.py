from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
%matplotlib inline
import matplotlib.pyplot as plt
image_index = 35
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()
print(x_train.shape)
print(x_test.shape)
print(y_train[:image_index + 1])

# save input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train /= 255
x_test /= 255

from keras.utils import to_categorical
num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
     activation='relu',
     input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

batch_size = 128
epochs = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("test_model.h5")

import imageio
import numpy as np
from matplotlib import pyplot as plt

im = imageio.imread("data:image/jpeg;base64,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")

gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

# reshape the image
gray = gray.reshape(1, img_rows, img_cols, 1)

# normalize image
gray /= 255

# load the model
from keras.models import load_model
model = load_model("test_model.h5")

# predict digit
prediction = model.predict(gray)
print(prediction.argmax())

