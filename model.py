import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Activation, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPool2D
import sklearn
from sklearn.model_selection import train_test_split
lines = []

with open('training/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

images = []
measures = []
for line in lines:
    filename = line[0].split('/')[-1]
    path = 'training/IMG/' + filename
    img = cv2.imread(path)
    images.append(img)
    measures.append(line[3])
    images.append(cv2.flip(img, 1))
    # print(float(line[3]))
    # print(type(float(line[3])))

    measures.append(float(line[3]) * (-1.0))


X_train = np.array(images)
# X_train = X_train / 255.0 - 0.5
y_train = np.array(measures)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'training/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                images.append(cv2.flip(img, 1))
                angles.append(float(center_angle) * (-1.0))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print(len(X_train))
            yield sklearn.utils.shuffle(X_train, y_train)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,
                 input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Flatten())
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print(len(train_samples))
print(len(validation_samples))

history_object = model.fit_generator(train_generator, steps_per_epoch=len(
    train_samples) // 32 + 1, validation_data=validation_generator, validation_steps=len(validation_samples) // +1, verbose=1, epochs=7)

print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# model.fit_generator(train_generator, samples_per_epoch=len(
# train_samples)*2, validation_data=validation_generator,
# nb_val_samples=len(validation_samples)*2,verbose=2, nb_epoch=7)

model.save('model.h5')
