## Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/non-vehicle.png
[image11]: ./output_images/vehicle.png
[image2]: ./output_images/car_hog_0.png
[image21]: ./output_images/car_hog_1.png
[image22]: ./output_images/car_hog_2.png
[image23]: ./output_images/notcar_hog_0.png
[image24]: ./output_images/notcar_hog_1.png
[image25]: ./output_images/notcar_hog_2.png
[image3]: ./output_images/grid1.png
[image31]: ./output_images/grid2.png
[image32]: ./output_images/grid3.png
[image33]: ./output_images/grid4.png
[image4]: ./output_images/test_0_out.png
[image41]: ./output_images/test_1_out.png
[image42]: ./output_images/test_2_out.png
[image43]: ./output_images/test_3_out.png
[image44]: ./output_images/test_4_out.png
[image45]: ./output_images/test_5_out.png

[image5]: ./output_images/frame1.png
[image51]: ./output_images/frame2.png
[image52]: ./output_images/frame3.png
[image53]: ./output_images/frame4.png
[image54]: ./output_images/frame5.png
[image55]: ./output_images/frame6.png


[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the IPython notebook and function `extract_features_hog` in the file called `./codes/helper.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image11]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on a car image and a non-car image:

![alt text][image2]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and many of the results are very close to each other. Please refer to the ipython notebook for the full grid searched HOG pictures.

Finally I decided to use

* cspace = `YCrCb`
* Orient = 9
* Pix_per_cell = 8
* Cell_per_block = 2
* hog_channel = `ALL`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC using default parameters. Without tuning any parameters, the testing accuracy was 98.9%, which is amazing.

For feature extraction, please refer to the `extract_image_features` function in the `./codes/helper.py` file. This function gives the option to extract any of the three features:

* HOG features using `get_hog_features`
* Histogram features using `color_hist`
* Spatial feature using `bin_spatial`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Please refer to the `find_vehicles` function in the `./codes/helper.py` file. 

To be brief, I use 3 different sizes sliding window search (ref `slide_window` function), with:

* x_start_stop: `[None, None]`,  `[None, None]`,  `[None, None]`
* y_start_stop: `[395, 650]`, `[395, 650]`, `[395, 650]`
* xy_window: `(128, 96)`, `(96, 96)`, `(48, 48)`
* xy_overlap: `(0.5, 0.5)`, `(0.8, 0.8)`, `(0.45, 0.45)`

![alt text][image3]
![alt text][image31]
![alt text][image32]
![alt text][image33]

I then feed there 3 different sized windows into the `search_windows` function, together with the pre-trained `svc` and `x_scaler`:

1. Iterate over all windows
2. Extract the test window from original image & resize
3. Extract features for that window using `extract_image_features`
4. Scale extracted features to be fed to classifier
5. Predict using SVC
6. If prediction == 1, append this window to the results

Finally, I drew there "hot" windows onto the image and return.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image4]
![alt text][image41]
![alt text][image42]
![alt text][image43]
![alt text][image44]
![alt text][image45]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* It takes 2+ hours to process the <1 min video. This cannot be done in real time, which is somehow useless. I would recommend using **Faster-RCNN** for real time object detection (e.g. [YOLO](https://pjreddie.com/darknet/yolo/)).
* 

