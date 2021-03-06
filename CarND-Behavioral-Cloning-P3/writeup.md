# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The starting point is the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with the following architecture, with all activation function being `ReLu`. (`model.py line 159-199`)

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/nVidia_model.png)


#### 2. Attempts to reduce overfitting in the model

Instead of using `dropout`, we use `l2` regularization `=0.001` on `W`s in both the convolution and dense layers.

To further avoid overfitting, we randomly distort the training images via (`model.py line 100-129`)

* random brightness
* random shadow
* randomly shift horizon

#### 3. Model parameter tuning

The model used an `adam` optimizer with `lr=1e-4`, so the learning rate was not tuned manually (`model.py line 198`).


#### 4. Appropriate training data

We use the default training data provided by Udacity.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with cropping only the road part.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to construct a CNN that reads the road image picture and give the steering angle.

My first step was to use a convolution neural network model similar to the nVidia model. I thought this model might be appropriate because its architecture was carefully chose and proved.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I think I should not only modify the model architecture (i.e. introducing regularization), but also generalize & augment the training data. 

To augment training data, I use random distortion on top of the images. I also use the left/right camera with `+/- 0.25` correction. Moreover, I randomly vertically revert the pictures.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as the left turn after the bridge. It seems the car just go straight into the side track. To improve the driving behavior in these cases, I analyze the training set. I notice that most of the data's corresponding angle are centered at `0`, which provide limited information on how to behave when the car are not on the center of the track.

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/data_old.png)

I re-weighted the training set by dropping and achieved a better distributed histogram:

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/data_new.png)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py lines 159-199`) consisted of a convolution neural network:

1. Input of size (66, 200, 3) (by resizing the image after cropping & distorting.)
2. Normalization by `x / 127.5 -1`.
3. Three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride, `valid` padding, `ReLu` activation after each Conv layer, `W_regularizer` being `l2(0.001)`.
4. Two 3x3 convolution layers (output depth 64, and 64), each with 1x1 stride, `valid` padding, `ReLu` activation after each Conv layer, `W_regularizer` being `l2(0.001)`.
5. Flatten layer
6. Three fully connected layers (depth 100, 50, 10), Relu activation, `W_regularizer` being `l2(0.001)`.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the middle of the lane when they are about to go outside of the trail. These images show what a recovery looks like:

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/left_2017_09_22_10_48_01_917.jpg)
![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/right_2017_09_22_10_47_56_544.jpg)


To augment the data sat, I also flipped images and angles thinking that this would help the model to get more dataset to train. Moreover, I also crop and distort the image to make the model less overfitting:

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/data_pre_crop.png)
![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/data_post_crop.png)

Later I re-weight the sample so that the data with high angle values in absolute value will get more weights during the training.

I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 150 as evidenced by the loss function below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/examples/loss.png)

