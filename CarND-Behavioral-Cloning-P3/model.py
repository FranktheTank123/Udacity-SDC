import argparse
import os
import sys
import csv, copy
import base64
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import sklearn
from sklearn.utils import shuffle

###
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam


# define global variable
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_PATH, 'data')



def extract_data_from_csv(csv_file, correction):
    """
    Extract imange path and angles.

    For center images, we don't touch the angles.
    For left images, we add correction to angles.
    For right images, we subtract correction to angles.
    """
    image_paths, angles = [], []
    with open(os.path.join(DATA_PATH, csv_file)) as csvfile:
        reader = list(csv.reader(csvfile, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
        for row in reader[1:]: # first row is column name
            # skip it if ~0 speed - not representative of driving behavior
            if float(row[6]) < 0.1 :
                continue
            # get center image path and angle
            image_paths.append(row[0])
            angles.append(float(row[3]))
            # get left image path and angle
            image_paths.append(row[1])
            angles.append(float(row[3])+correction)
            # get left image path and angle
            image_paths.append(row[2])
            angles.append(float(row[3])-correction)
            
    image_paths = np.array(image_paths)
    angles = np.array(angles)
    return image_paths, angles

def clean_data(image_paths, angles):
    num_bins = 23
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)

    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/target))

    remove_list = []
    for i in range(len(angles)):
        for j in range(num_bins):
            if angles[i] > bins[j] and angles[i] <= bins[j+1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)
    image_paths = np.delete(image_paths, remove_list, axis=0)
    angles = np.delete(angles, remove_list)
    return image_paths, angles

def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are 
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def random_distort(img):
    ''' 
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return new_img.astype(np.uint8)

def generator(image_paths, angles, batch_size=32, **kwargs):
    num_samples = len(image_paths)
    while 1: # Loop forever so the generator never terminates
        image_paths, angles = shuffle(image_paths, angles)
        for offset in range(0, num_samples, batch_size):
            batch_samples = (image_paths[offset:offset+batch_size], angles[offset:offset+batch_size])

            images_batch = []
            angles_batch = []
            for image_path, angle in batch_samples:

                img = cv2.imread(os.path.join(DATA_PATH, image_path))
                img = preprocess_image(img)
                if not validation_flag:
                    img = random_distort(img)

                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)
                    angle *= -1

                images_batch.append(img)
                angles_batch.append(angle)

            # trim image to only see section with road
            X_train = np.array(images_batch)
            y_train = np.array(angles_batch)
            yield shuffle(X_train, y_train)

def get_model():
    model = Sequential()

    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66,200,3)))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
    #model.add(Dropout(0.50))

    # Add a fully connected output layer
    model.add(Dense(1))

    # Compile and train the model, 
    #model.compile('adam', 'mean_squared_error')
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

def plot_error(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main(correction, batch_size, nb_epoch):


    # load samples
    image_paths, angles = extract_data_from_csv('driving_log.csv', correction=correction)

    # re-weight data
    image_paths, angles = clean_data(image_paths, angles)

    image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles, test_size=0.05, random_state=1111)
    train_gen = generator(image_paths_train, angles_train, batch_size=batch_size)
    val_gen = generator(image_paths_test, angles_test, batch_size=batch_size)

    model = get_model()

    history_object = model.fit_generator(train_gen, 
    					samples_per_epoch=len(image_paths_train), 
    					validation_data=val_gen,
                		nb_val_samples=len(image_paths_test), 
                		nb_epoch=nb_epoch)

    plot_error(history_object)


    model.save('model.h5')

if __name__ == '__main__':
    correction = 0.25
    batch_size = 64
    nb_epoch = 100

    main(correction=correction, batch_size=batch_size, nb_epoch=nb_epoch)


