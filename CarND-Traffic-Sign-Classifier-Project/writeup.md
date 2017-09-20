# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `numpy` library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram with `Seaborn` to show a smoothed distribution of the labels.

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/solution/traffic_data_vis.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color of the traffic sign seldom means anything.

Here is an example of a traffic sign image before and after grayscaling.

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/solution/color_example.png)

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/solution/grey_example.png)

As a last step, I normalized the image data using `(pixel - 128)/ 128` approximation because it's better to make the input dimensions with the mean 0 and small std.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description| 
|:---------------------:|:---------------------------------------------:| 
| Input  | 32x32x1 RGB image | 
| Convolution 3x3     	| 20 filters, 1x1 stride, `VALID` padding, outputs 30x30x20 	|
| RELU | |
| Dropout | keep_prob = 0.5|
| Max pooling	| 3x3 kernel, 1x1 stride,  outputs 28x28x20 |
| Convolution 3x3     	| 50 filters, 1x1 stride, `VALID` padding, outputs 26x26x50 	|
| RELU | |
| Dropout | keep_prob = 0.5|
| Max pooling	| 2x2 kernel, 2x2 stride,  outputs 13x13x50 |
| Convolution 4x4     	| 100 filters, 1x1 stride, `VALID` padding, outputs 10x10x100 	|
| RELU | |
| Dropout | keep_prob = 0.5|
| Max pooling	| 2x2 kernel, 2x2 stride,  outputs 5x5x100 |
|Flatten Layer| outputs 2500x1|
| Fully connected		| outputs 1024x1|
| RELU | |
| Dropout | keep_prob = 0.5|
| Fully connected		| outputs 521x1|
| RELU | |
| Dropout | keep_prob = 0.5|
| Fully connected		| outputs 128x1|
| RELU | |
| Dropout | keep_prob = 0.5|
| Fully connected		| outputs 43x1|
|`softmax_cross_entropy_with_logits`||
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model:

1. **EPOCH**: I increase the EPOCH number to 50
3. **Learning Rate**: Learning rate was optimal at 0.001. Any change from here degraded the performance.
4. **Optimizer**: I tried RMSProp and Adam, and Adam seems better.
5. **Regularization**: I used dropout = 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 99.3%
* validation set accuracy of 95.1% 
* test set accuracy of 93.7%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
    * At first I chose LeNet-5 because I have a sample code.
* What were some problems with the initial architecture?
    * The model was pretty shallow, not getting me good OOS results. (**underfitting**)
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * As mentioned about, since the original LeNet-5 was underfitting, I should start with expand the parameters, by having extra conv and FC layers with larger size. I didn't change the ReLu activation function as it's the state-of-art.
* Which parameters were tuned? How were they adjusted and why?
    * I tried to increase the batch size to 256, 512 etc. but result underperformed. Therefore, I decided to keep batch size as 128.   
    * I tried to change learning rate from 0.0001 to 0.01, but 0.001 seems to be the best.
    * I tried Adam and RMSProp, and Adam seems better.
    * I also tried droupout with keep_prob = 0.5 to reduce overfitting of the bigger net.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * CNN works very well for image classification, as the shared weights will reduce parameters space. Moreover, it doesn't matter whether there was a shift/rotation in the image.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/new_images/Double_curve.png)
![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/new_images/SpeedLimit30.png)
![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/new_images/SpeedLimit60.png) 
![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/new_images/SpeedLimit80.png) 
![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/new_images/slippery_road.png)

Personally I believe numbers are hard to identify because they are not rotation-invariant. 6 and 9 are different, but hard for CNN to tell if there are too much weight sharing. `Double curve` and `slippery road` are also hard to learn because there are less dataset in the training, which means the model is less familiar with these picture.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image	| Prediction| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)| Speed limit (60km/h)| 
|Speed limit (30km/h)| Keep right |
| Double curve	| Right-of-way at the next intersection|
| Speed limit (80km/h)	| Speed limit (80km/h) |
| Slippery road | Slippery road|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The accuracy is far less than the test accuracies.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text](https://github.com/FranktheTank123/Udacity-SDC/blob/master/CarND-Traffic-Sign-Classifier-Project/solution/results.png)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Pass.

