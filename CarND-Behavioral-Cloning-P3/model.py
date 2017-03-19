import csv
import cv2
import numpy as np
import sklearn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
import math
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline


samples = []

csv_path = ['data/data/driving_log.csv', 'data/ds1/driving_log.csv']

for j in range(2):
    if j==2:
        # 0 = my own data, 1 = Udacity supplied data , any other no = use both dataset
        print('not using dataset ', j)
        continue
    with open(csv_path[j]) as csvfile:
        #csvlines = csvfile.readlines()  # if we use this method then comment below 3 lines and replace samples by csvlines
        reader = csv.reader(csvfile)
        for line in reader:
             # skip it if ~0 speed - not representative of driving behavior
            if float(line[6]) < 0.1 :
                continue
            if float(line[3])>0.01 or float(line[3])<-0.01:
                samples.append(line)
            else:
                select_prob = np.random.random()
                if select_prob > 0.80:
                    samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
a= np.array(samples)

# print a histogram to see which steering angle ranges are most overrepresented
a_angles = a[:,3].astype(np.float)
num_bins = 25
avg_samples_per_bin = len(a)/num_bins
hist, bins = np.histogram(a_angles, num_bins)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(a_angles), np.max(a_angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')

hist = np.concatenate((hist, [1]))  # because no of bins are 26 (0 to 25), but no of hist were 25.(see print(bins, hist) below)
print(bins, hist)


def process_image(image):
    
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (64, 64))
    
    #image2 = np.zeros(image.shape, dtype='u1')
    #image2 = brightness_process_image(image, image2)
    
    return image




threshold = avg_samples_per_bin * 0.5


def Augmentation(row, car_images, steering_angles):
    steering_center = float(row[3])
    digitized_center = np.digitize(steering_center,bins)-1   #bin number for steering_center angle for this row
    
    # create adjusted steering measurements for the side camera images
    correction = 0.20     # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    digitized_left = np.digitize(steering_left,bins)-1    #bin number for steering_left angle for this row
    digitized_right = np.digitize(steering_right,bins)-1   #bin number for steering_right angle for this row
    
    select_prob = np.random.random()
    
    # read in images from center, left and right cameras
    path = 'data/data/' # fill in the path to your training IMG directory
    
    
    
    # It balances histogram, if image is in the bin having no of images greater than threshold ....(see above 2 cells)
    #1./(hist[digitized_center]/threshold) is equivalent to  threshold/hist[digitized_center] (i.e 100/500 = 0.2 = keep_probs)
    if hist[digitized_center] > threshold:   #hist[digitized_center] will give no of images in that bin
        keep_probs_center = 1./(hist[digitized_center]/threshold)
    else:
        keep_probs_center = 1
    if select_prob > (1-keep_probs_center):
        img_center =  process_image(np.asarray(Image.open(path+row[0])))
        img_center_flipped = np.fliplr(img_center)
        car_images.append(img_center)
        car_images.append(img_center_flipped)
        steering_angles.append(steering_center)
        steering_angles.append(-steering_center)
        
    if hist[digitized_left] > threshold:
        keep_probs_left = 1./(hist[digitized_left]/threshold)
    else:
        keep_probs_left = 1
    if select_prob > (1-keep_probs_left) and steering_left<0.99:
        img_left =  process_image(np.asarray(Image.open(path+row[1])))
        img_left_flipped = np.fliplr(img_left)
        car_images.append(img_left)
        car_images.append(img_left_flipped)
        steering_angles.append(steering_left)
        steering_angles.append(-steering_left)
        
    if hist[digitized_right] > threshold:
        keep_probs_right = 1./(hist[digitized_right]/threshold)
    else:
        keep_probs_right = 1
    if select_prob > (1-keep_probs_right) and steering_right>-0.99:
        img_right =  process_image(np.asarray(Image.open(path + row[2])))
        img_right_flipped = np.fliplr(img_right)
        car_images.append(img_right)
        car_images.append(img_right_flipped)
        steering_angles.append(steering_right)
        steering_angles.append(-steering_right)



def generator(samples, batch_size=64):
    num_samples = int(len(samples)/3)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []

            for row in batch_samples:
                Augmentation(row, car_images, steering_angles)

            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            #print(X_train.shape, y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)




# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)



model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(64,64,3)))
# layer 1 output shape is 32x32x32
model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
model.add(ELU())
# layer 2 output shape is 15x15x16
model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(ELU())
model.add(Dropout(.4))
model.add(MaxPooling2D((2, 2), border_mode='valid'))
# layer 3 output shape is 12x12x16
model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
model.add(ELU())
model.add(Dropout(.4))
# Flatten the output
model.add(Flatten())
# layer 4
model.add(Dense(1024))
model.add(Dropout(.3))
model.add(ELU())
# layer 5
model.add(Dense(512))
model.add(ELU())
# Finally a single output, since this is a regression problem
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train,validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*2, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples)*2, nb_epoch=15)         #https://keras.io/models/sequential/#fit_generator

model.save('model_test.h5')