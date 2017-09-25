
import numpy as np
import cv2
import pickle


def pipeline(img, s_thresh=(170, 255), sx_thresh=(30, 150)):
    """pipeline, single images."""
    img = np.copy(img)
    
    # Convert to HLS color space and separate the l & s channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold HLS color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    final_result = np.zeros_like(sxbinary)
    final_result[(sxbinary == 1) |(s_binary ==1)] =1

    return color_binary, final_result

def warper(img, src, dst):
    """Perspective Transform"""
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def lines_sanity_check(lens, left_fit, right_fit, left_curverad, right_curverad):
    """Make sure curves are reasonable."""
    try:
        dist_thresh = (650, 750)    # Distance of left and right lane in pixels
        roc_max_thresh = 10000      # Threshold for straight lines
        roc_diff_thresh = (0, 1000) # Threshold for curved lines

        # Distance Check
        left = left_fit[0]*lens**2 + left_fit[1]*lens + left_fit[2]
        right = right_fit[0]*lens**2 + right_fit[1]*lens + right_fit[2]
        base_diff = np.abs(left - right)
        base_diff_check = (base_diff > dist_thresh[0]) & (base_diff < dist_thresh[1])
        
        # ROC check
        roc_diff = np.abs(right_curverad - left_curverad)
        roc_max_check = (left_curverad > roc_max_thresh) | (right_curverad > roc_max_thresh)
        roc_diff_check = (roc_diff > roc_diff_thresh[0]) & (roc_diff < roc_diff_thresh[1])
        roc_check = roc_max_check | roc_diff_check

        # Parallel Check. Checking if they are of same sign or not
        coeff1_sign = left_fit[0]*right_fit[0]
        coeff2_sign = left_fit[1]*right_fit[1]
        parallel_check = (coeff1_sign > 0) & (coeff2_sign > 0) 

        # Assersion
        if base_diff_check and roc_check and parallel_check:
            return True
        else:
            return False  
    except:
        return False


class Line():
    def __init__(self):

        # load distortion correction
        with open('src/mtx.pkl', 'rb') as temp:
            self.mtx = pickle.load(temp)    
        with open('src/dist.pkl', 'rb') as temp:
            self.dist = pickle.load(temp)

        self.SRC_COORDS = np.float32([(203,720),(1127,720),(695,460),(585,460)])
        self.DEST_COORDS = np.float32([(320,720),(960,720),(960,0),(320,0)])


        ## Smoothing parameters below
        # was the line detected in the last iteration?
        self.detected = False 
        self.N = 5

        # x values of the last n fits of the line
        self.last_xfitted_left = [] 
        self.last_xfitted_right = []

        # coefficients of the last n fits of the line
        self.last_fit_left = []
        self.last_fit_right = []

        # average x values of the fitted line over the iterations
        self.bestx_left = []    
        self.bestx_right = []
        # polynomial coefficients averaged over the iterations
        self.best_fit_left = []  
        self.best_fit_right = []

        # polynomial coefficients for the most last fit
        self.current_fit_left = []  
        self.current_fit_right = []

        #radius of curvature of the line in some units
        self.radius_of_curvature_left = None 
        self.radius_of_curvature_right = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        self.shape = None
        self.count_faulty_images = 0
        
        self.position = None
    
    # Function for obtaining ROC 
    def get_radius_of_curvature(self, nonzerox,nonzeroy,left_lane_inds,right_lane_inds,left_fit,right_fit):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        left_curverad = ((1 + (2*left_fit[0]*self.shape[0] + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*self.shape[0] + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad
    
    def detect_lane_lines(self, binary_warped, previous = True):
        # Function to detect lane lines. Previous = True if using previous images
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)
        minpix = 50
        margin = 50
        left_lane_inds = []
        right_lane_inds = []
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        if previous == True:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            base_y = self.shape[1]-1
            
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                leftx_current = self.current_fit_left[0]*(win_y_high -1)**2 + self.current_fit_left[1]*(win_y_high -1) + self.current_fit_left[2]
                rightx_current = self.current_fit_right[0]*(win_y_high -1)**2 + self.current_fit_right[1]*(win_y_high -1) + self.current_fit_right[2]
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
        else:
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, laster next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)    
        except:
            left_fit = []
            right_fit = []
        return out_img, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds
       
    def _image_preprocessing(self, image):
        # image preprocessing
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.shape = image.shape
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        pipeline_result, final_result = pipeline(image)                  # gradient thresholding
        warped = warper(final_result, self.SRC_COORDS, self.DEST_COORDS) # perspective transform
        return warped

    def _calc_dist_from_center(self):
        # meters from center
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        screen_middel_pixel = self.shape[1]/2
        y_eval = self.shape[0]
        left_lane_pixel =  self.best_fit_left[0]*y_eval**2 + self.best_fit_left[1]*y_eval + self.best_fit_left[2]
        right_lane_pixel = self.best_fit_right[0]*y_eval**2 + self.best_fit_right[1]*y_eval + self.best_fit_right[2]
        car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
        screen_off_center = screen_middel_pixel-car_middle_pixel
        meters_off_center = xm_per_pix * screen_off_center
        self.position = meters_off_center

    def _append_and_smoothing(self):
        ploty = np.linspace(0, self.shape[0]-1, self.shape[0])
        left_ = self.current_fit_left[0]*ploty**2 + self.current_fit_left[1]*ploty + self.current_fit_left[2]
        right_ = self.current_fit_right[0]*ploty**2 + self.current_fit_right[1]*ploty + self.current_fit_right[2]

        self.last_xfitted_left.append(left_)
        self.last_xfitted_right.append(right_)
        self.last_fit_left.append(self.current_fit_left)
        self.last_fit_right.append(self.current_fit_right)
    
        if not len(self.last_xfitted_left):
            # This is because first image is different from rest of the images
            self.bestx_left.append(left_)
            self.bestx_right.append(right_)
            self.best_fit_left = self.current_fit_left
            self.best_fit_right = self.current_fit_right

        else: # we only we a weighted MV of the last N segments.
            if len(self.last_xfitted_left) > self.N:
                self.last_xfitted_left = self.last_xfitted_left[1:]
                self.last_xfitted_right = self.last_xfitted_right[1:]
                self.last_fit_left = self.last_fit_left[1:]
                self.last_fit_right = self.last_fit_right[1:]
            
            # Gettting the new bestx
            self.bestx_left = np.mean(self.last_xfitted_left, axis=0).astype(int).tolist()
            self.bestx_right = np.mean(self.last_xfitted_right, axis=0).astype(int).tolist()
            # Getting the new best_fit
            self.best_fit_left = np.mean(self.last_fit_left, axis=0).tolist()
            self.best_fit_right = np.mean(self.last_fit_right, axis=0).tolist()

    def _get_road_image(self, warped, image):
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx  = self.best_fit_left[0]*ploty**2  + self.best_fit_left[1]*ploty  + self.best_fit_left[2]
        right_fitx = self.best_fit_right[0]*ploty**2 + self.best_fit_right[1]*ploty + self.best_fit_right[2]

        Minv = cv2.getPerspectiveTransform(self.DEST_COORDS, self.SRC_COORDS)
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Get the distance from center for the image
        self._calc_dist_from_center()
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (self.shape[1], self.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        cv2.putText(result, 'Radius of Curvature :' + str(round(self.radius_of_curvature_left,0)) + "," + str(round(self.radius_of_curvature_right,0)), (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        cv2.putText(result, 'Distance from Center :' + str(round(self.position,2)) , (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        return result

    def iter_once(self, image):
        warped = self._image_preprocessing(image)

        out_img, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.detect_lane_lines(warped, self.detected)
        try:
            leftcurverad, rightcurverad = self.get_radius_of_curvature(nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit)
            sanity_check = lines_sanity_check(self.shape[0]-1, left_fit, right_fit, leftcurverad, rightcurverad)
        except: 
            leftcurverad, rightcurverad = self.radius_of_curvature_left, self.radius_of_curvature_right
            sanity_check = False

        if sanity_check or len(self.last_xfitted_left)==0: # either pass sanity check, or is the first frame.
            self.detected =True
        else: # do another round with self.detected = False
            if self.count_faulty_images < 3:
                self.detected= False
                self.count_faulty_images += 1
                return self._get_road_image(warped, image)
            else:
                self.count_faulty_images = 0
                out_img, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = self.detect_lane_lines(warped,False)
                try:
                    leftcurverad,rightcurverad = self.get_radius_of_curvature(nonzerox,nonzeroy,left_lane_inds,right_lane_inds,left_fit,right_fit)
                    sanity_check2 = lines_sanity_check(self.shape[0]-1, left_fit, right_fit, leftcurverad, rightcurverad)
                except:
                    leftcurverad, rightcurverad = self.radius_of_curvature_left, self.radius_of_curvature_right
                    sanity_check2 = False
                if sanity_check:
                    self.detected =True
                else:
                    self.detected = False
                    self.count_faulty_images += 1
                    return self._get_road_image(warped, image)

        self.current_fit_left = left_fit
        self.current_fit_right = right_fit
        self.radius_of_curvature_left = leftcurverad
        self.radius_of_curvature_right = rightcurverad
        # append results and smoothing
        self._append_and_smoothing()

        # Generation of Final Image
        final_image = self._get_road_image(warped, image)

        return final_image
    



