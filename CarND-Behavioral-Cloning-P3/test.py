import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('data/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]	#column 0 in csv
	filename = source_path.split('\\')[-1]	#take filename from whole path
	current_path = 'data/data/IMG_test/' + filename
	image = cv2.imread(current_path)
	image_flipped = np.fliplr(image)
	images.append(image)
	images.append(image_flipped)
	measurement = float(line[3])	#Coulumn 3 : Steering angle
	measurement_flipped = -measurement  #Steering angle for flipped image
	measurements.append(measurement)
	measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())		#or can be MaxPooling2D((2, 2))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(75))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,validation_split=0.2, shuffle=True)

model.save('model_test.h5')
