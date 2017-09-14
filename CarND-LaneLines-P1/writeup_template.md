# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 

1. Convert the images to grayscale via `grayscale`
2. Apply Gaussian smoothing via `gaussian_blur`
3. Apply Canny Edge Detectiom via `canny`
4. Define and mask region of interest via `draw_roi_lines` and `region_of_interest`
5. Hough transform the masked image with pre-specified parameters via `hough_lines`.

    ```python
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5 #minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
```

6. Drawing lines on the original image via `weighted_img`


In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function to `draw_lines_mod` by:

* Grouping right and left lanes according to their slopes. Here I define the right lane's slope within (-10, -0.5), and left lane's slope with (0.5, 10).
* For each line, I represent it using its middle point (so that I just need 1 tuple instead of 2)
* For each lane group, sort the lines from bottom to top according to their y-axis (largest first). The reason is that the more bottom the lines are, the more closer they are to the camera, and the more accurate the line detection will be relatively.
* For each group, starting from bottom, calculate the "pseudo-average" slope between line 0 and line `i`, for `i` in `range(0, len(lines))`, until the averaged line's slope go beyond the slope boundries (0.5, 10) in absolute sense.
* After we got the slope for each lane, we use it to extend **the bottom line** till the bottom and till the top of region of interest.
* Moreover, we introduce a smoothing parameter `exp_w=0.2`, which to make sure the currently slope not being too far away from the slope in the last frame. This is implemented using a `global dictionary`. 

![alt text][test_images_output/test.png]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the video changes, I need manually define/tune the region of interest vertices and the low/high threshold of Canny Edge Detection. This becomes more impractical when we have tons of videos to lable.

Another shortcoming could be the when lanes are not really alines. They can be curly, dashed, double solid, solid, etc. Moreover, the currently pipeline does not differentiate between different types of lane, which could less meaningful.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to classify different types of lane.

Another potential improvement could be to make the lane curly.


