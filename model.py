import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten,  Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
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


def add_flip_image(images, angles, img, angle, valid=False):
    images.append(img)
    angles.append(angle)
    if not valid:
        images.append(cv2.flip(img, 1))
        angles.append(float(angle) * (-1.0))


def generator(samples, batch_size=32, valid=False):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = 'training/IMG/'
                name = path + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                steering_center = float(batch_sample[3])
                add_flip_image(images, angles, center_image, steering_center)
                if not valid:
                    correction = 0.2  # this is a parameter to tune
                    steering_left = steering_center + correction

                    img_left = cv2.imread(
                        path + batch_sample[1].split('/')[-1])
                    # print(type(img_left))
                    add_flip_image(images, angles, img_left, steering_left)

                    steering_right = steering_center - 0.22

                    img_right = cv2.imread(
                        path + batch_sample[2].split('/')[-1])
                    # print(type(img_right))

                    add_flip_image(images, angles,  img_right, steering_right,)

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
model.add(Dropout(0.5))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.5))
model.add(Dense(50, kernel_regularizer=l2(0.0001)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)
train_batch_size = 16
valida_batch_size = 96
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=train_batch_size)
validation_generator = generator(
    validation_samples, batch_size=valida_batch_size, valid=True)

# print(len(train_samples))
# print(len(validation_samples))

history_object = model.fit_generator(train_generator, steps_per_epoch=len(
    train_samples) // train_batch_size + 1, validation_data=validation_generator, validation_steps=len(validation_samples) // valida_batch_size + 1, verbose=1, epochs=6)


# model.fit_generator(train_generator, samples_per_epoch=len(
# train_samples)*2, validation_data=validation_generator,
# nb_val_samples=len(validation_samples)*2,verbose=2, nb_epoch=7)

model.save('model.h5')
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
