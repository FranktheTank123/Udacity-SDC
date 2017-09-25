## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistorted.jpg "Undistorted"
[image2]: ./output_images/Undistorted_test.jpg "Undistorted_test"
[image3]: ./output_images/color_binary.jpg "Binary Example"
[image4]: ./output_images/perspective_1.jpg "Warp Example1"
[image41]: ./output_images/perspective_2.jpg "Warp Example2"
[image51]: ./output_images/source_img.png "Fit Visual"
[image52]: ./output_images/perspective.png "Fit Visual2"
[image53]: ./output_images/detected.png "Fit Visual3"
[image54]: ./output_images/lanes.png "Fit Visual4"
[image6]: ./output_images/test1.png "Output1"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3rd code cell of the IPython notebook located in `./Advanced Lane Finding workbook.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of HLS color and gradient thresholds to generate a binary image (refer to the `pipeline` function in the notebook).  Here's an example of my output for this step. For thresholds, I tested and finally chose define: 

* Sobel X threshold: `sx_thresh=(30,150)`
* HLS's S threshold: `s_thresh=(170, 255)`

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the ipython notebook.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image41]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane lines are detected in the function `detect_lane_lines` in the `./codes/Line.py` line 147-231. I used the polynomial fit method from the lectures to detect the lane lines.

Here are the steps I take for detecting lane lines:

* Parameters:
    *   Num windows = 9
    *   Minpix = 50
    *   Margin = 50
* Get the histogram of the bottom half of the image and obtain the peaks for left and right.
* Use the peaks pixels on the left and right of the image as a starting point for lane search.
* Obtain all the non-zero pixels within the margin (used 50) of the center pixels.
* For next window, if you obtain more than minpix no of pixels for either left or right lane, use the mean of those pixels as a starting point for center of next window.
* Use the non zero pixels obtained on the left and right to fit the left and right quadratic curves respectively.

When we process the video, I will save the frame. For the next image, I do a targeted search based on the lane lines in previous frame. For search of center pixels in each window, I use the polynomial coefficients from previous frame to obtain the center pixel for new frame. Rest of the procedure remains the same as above.

![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Please refer to the `get_radius_of_curvature` function defined in the `./codes/Line.py` line 131-145. I simply refer to the codes provided from #35 - Measuring Curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the whole pipeline in the `Line` class in the `./codes/Line.py`. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further:

* There are too many hand-tuned parameters, which makes the pipeline hard to generalize. For instance, my pipeline did not work on the harder video.
* The pipeline fails when there are shades, strong lights, curvy road (when part of the road was outside of the region of interest, which makes it difficult to extract the real curvature.), multiple lanes, etc.

In order to make the pipeline more robust, I would recommend to approach the problem from a deep learning way, where the model can utilize all part of the image, which can deal with all situations (like the human does by only seeing the scene from the window/camera).

