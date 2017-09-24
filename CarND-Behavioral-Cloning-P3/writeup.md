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

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Behavioral-Cloning-P3/nVidia_model.png)


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

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as the left turn after the bridge. It seems the car just go straight into the side track. To improve the driving behavior in these cases, I analyze the training set. I notice that most of the data are 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py lines 159-199`) consisted of a convolution neural network with the following layers and layer sizes ...


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


