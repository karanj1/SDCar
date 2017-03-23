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
from keras import regularizers
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline


samples = []
#csv_path = ['data/data/driving_log.csv', 'data/ds1/driving_log.csv', 'data/DSReverse/driving_log.csv']
csv_path = ['data/data/driving_log.csv', 'data/ds1/driving_log.csv', 'data/DSReverse/driving_log.csv', \
            'data/data/data_pat/track1_central/driving_log.csv', 'data/data/data_pat/track1_recovery/driving_log.csv', \
            'data/data/data_pat/track1_recovery_reverse/driving_log.csv', 'data/data/data_pat/track1_reverse/driving_log.csv']
for j in range(7):
    if j==8:
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
            if (float(line[3])>0.01 and float(line[3])<0.99) or (float(line[3])>-0.99 and float(line[3])<-0.01):
                samples.append(line)
            else:
                select_prob = np.random.random()
                if select_prob > 0.85:
                    samples.append(line)
                
            #samples.append(line)
        
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
a= np.array(samples)


def read_image(row):
    
    steering_center = float(row[3])
    # create adjusted steering measurements for the side camera images
    offset=1.0 
    dist=10.0
    correction = offset/dist * 360/( 2*np.pi) / 25.0     #17# this is a parameter to tune
    
    # read in images from center, left and right cameras
    path = 'data/data/' # fill in the path to your training IMG directory
    
    camera = np.random.choice(['center', 'left', 'right'], p=[0.3,0.35,0.35])
    
    if camera == "center" :    
        img = np.asarray(Image.open(path+row[0]))
        steering = steering_center
    elif camera == "left" :    
        img = np.asarray(Image.open(path+row[1]))
        steering = steering_center + correction
    elif camera == "right" :    
        img = np.asarray(Image.open(path + row[2]))
        steering = steering_center - correction
    else:
         print ('Invalid camera or path :',camera, row )
    
    return img, steering


def transform_image(image,steering,ang_range,shear_range,trans_range):
# Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
# Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
# Shear
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    M = cv2.getAffineTransform(pts1,pts2)
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    steering +=dsteering
        
    image = cv2.warpAffine(image,Rot_M,(cols,rows))
    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    
    return image,steering
    
    
def crop_image(image, cam_angle=0.0, train = True):
    
    tx_lower=-20
    tx_upper=20
    ty_lower=-10
    ty_upper=10
    shape = image.shape
    steering = 0.0
    
    # note: numpy arrays are (row, col)!
    
    #tx and ty are random no of pixels # only for training data
    if train:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
        # the steering variable needs to be updated to counteract the shift 
        if tx != 0:
            steering = tx/(tx_upper-tx_lower)/3.0   #tx can be +ve or -ve (between -20 and 20)
    else:
        tx,ty=0,0   #for  validation data, turn randomness off

    
    #image cropping top:1/4th of height, bottom: 25px, left:20px, right:20px
    image = image[math.floor(shape[0]/4)+ty:shape[0]-25+ty, 20+tx:shape[1]-20+tx]
    image = cv2.resize(image, (64, 64))
    cam_angle += steering
    
    return image,cam_angle

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def flip_image(image,cam_angle):
    img_flipped = np.fliplr(image)
    cam_angle_flipped = -cam_angle
    
    return img_flipped,cam_angle_flipped


def Augmentation(row, car_images=[], steering_angles=[], augment=False):
    
    image, cam_angle = read_image(row)
    
    image, cam_angle = transform_image(image, cam_angle, 2, 10, 2)
    
    image, cam_angle = crop_image(image, cam_angle, train=augment)
    
    image = random_brightness(image)
    
    # add images and angles to data set
    car_images.append(image)
    steering_angles.append(cam_angle)
    
    coin=np.random.randint(0,2)
    if coin==0 or coin==1:    
        image_flipped, cam_angle_flipped = flip_image(image,cam_angle)
        # add images and angles to data set
        car_images.append(image_flipped)
        steering_angles.append(cam_angle_flipped)
    
    
    return car_images, steering_angles




######################################################################

def generator(samples, augment=True, batch_size=256):
    num_samples = int(len(samples)/2.7)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []

            for row in batch_samples:
                if augment==True:
                    car_images, steering_angles = Augmentation(row, car_images, steering_angles)
                else:
                    car_images, steering_angles = Augmentation(row, car_images, steering_angles)

            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            #print(X_train.shape, y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


'''Model design'''
model = Sequential()
# Normalize
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(64,64,3)))
#model.add(Convolution2D(3,1,1,border_mode='valid', name='conv0'))
# layer 1 output shape is 32x32x32
model.add(Convolution2D(16, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
model.add(ELU())
# layer 2 output shape is 15x15x16
model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="valid"))
model.add(ELU())
model.add(Dropout(.4))
model.add(MaxPooling2D((2, 2), border_mode='valid'))
# layer 3 output shape is 13x13x16
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
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

print(model.summary())


'''Model run'''
model.compile(loss='mse', optimizer=Adam(lr=1e-4))
#model.fit(X_train, y_train,validation_split=0.2, shuffle=True, nb_epoch=7)
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*2, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples)*2, nb_epoch=8, verbose=1)          #https://keras.io/models/sequential/#fit_generator

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')