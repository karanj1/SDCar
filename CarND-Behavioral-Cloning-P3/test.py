import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation

lines = []
with open('data/driving_log_test.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]	#column 0 in csv
	filename = source_path.split('\\')[-1]	#take filename from whole path
	current_path = 'data/IMG_test/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])	#Coulumn 3 : Steering angle
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,validation_split=0.2, shuffle=True)

model.save('model_test.h5')
