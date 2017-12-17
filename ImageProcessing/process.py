import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def calibrate(img, objpoints, imgpoints):
    '''
    calibrate the camera
    '''
    h, w = img.shape[:2]
    img_size = (w, h)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def undist(img, mtx, dist):
    '''
    undistort the image
    '''
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(20, 100), channel='gray'):
    '''
    absolute value of gradient in either direction x or y
    '''
    assert(len(thresh) == 2)
    thresh_min, thresh_max = thresh

    if channel == 'gray':
        mono = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif channel == 'l':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mono = hls[:,:,1]
    else:
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mono = hls[:,:,2]

    if orient == 'y':
        sobel = cv2.Sobel(mono, cv2.CV_64F, 0, 1, ksize=kernel)
    else:
        sobel = cv2.Sobel(mono, cv2.CV_64F, 1, 0, ksize=kernel)

    scaled = (np.abs(sobel) / np.max(np.abs(sobel)) * 255).astype(np.uint8)

    binary_output = np.zeros_like(scaled)
    binary_output[(scaled > thresh_min) & (scaled < thresh_max)] = 1
    return binary_output

def mag_threshold(img, kernel=3, thresh=(30, 100), channel='gray'):
    '''
    magnitude of gradient
    '''
    assert(len(thresh) == 2)
    thresh_min, thresh_max = thresh

    if channel == 'gray':
        mono = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif channel == 'l':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mono = hls[:,:,1]
    elif channel == 'b':
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mono = lab[:,:,2]
    else:
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mono = hls[:,:,2]

    sobel_x = cv2.Sobel(mono, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(mono, cv2.CV_64F, 0, 1, ksize=kernel)
    mag = np.sqrt(sobel_x**2 + sobel_y ** 2)

    scaled = (mag / np.max(mag) * 255).astype(np.uint8)

    binary_output = np.zeros_like(scaled)
    binary_output[(scaled > thresh_min) & (scaled < thresh_max)] = 1
    return binary_output

def dir_threshold(img, kernel=15, thresh=(0.7, 1.3), channel='gray'):
    '''
    direction of gradient
    '''
    assert(len(thresh) == 2)
    thresh_min, thresh_max = thresh

    if channel == 'gray':
        mono = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif channel == 'l':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mono = hls[:,:,1]
    elif channel == 'b':
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mono = lab[:,:,2]
    else:
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        mono = hls[:,:,2]

    sobel_x = cv2.Sobel(mono, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(mono, cv2.CV_64F, 0, 1, ksize=kernel)
    abs_x, abs_y = np.abs(sobel_x), np.abs(sobel_y)
    direction = np.arctan2(abs_y, abs_x)

    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh_min) & (direction <= thresh_max)] = 1
    return binary_output

def hls_select(img, thresh=(170, 255), channel='s'):
    assert(len(thresh) == 2)
    thresh_min, thresh_max = thresh

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    if channel == 'h':
        mono = hls[:,:,0]
    else:
        mono = hls[:,:,2]

    binary_output = np.zeros_like(mono)
    binary_output[(mono >= thresh_min) & (mono <= thresh_max)] = 1
    return binary_output

def lab_select(img, thresh=(170, 255), channel='b'):
    assert(len(thresh) == 2)
    thresh_min, thresh_max = thresh

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    mono = lab[:,:,2]

    binary_output = np.zeros_like(mono)
    binary_output[(mono >= thresh_min) & (mono <= thresh_max)] = 1
    return binary_output

def thresholding(img, ksizes, thresholds):
    assert(isinstance(ksizes, dict))
    assert(isinstance(thresholds, dict))

    if 'grad_ksize' in ksizes:
        grad_ksize = ksizes['grad_ksize']
    else:
        grad_ksize = 3
    if 'mag_ksize' in ksizes:
        mag_ksize = ksizes['mag_ksize']
    else:
        mag_ksize = 9
    if 'dir_ksize' in ksizes:
        dir_ksize = ksizes['dir_ksize']
    else:
        dir_ksize = 15

    if 'grad_thresh' in thresholds:
        grad_thresh = thresholds['grad_thresh']
    else:
        grad_thresh = (30, 100)
    if 'mag_thresh' in thresholds:
        mag_thresh = thresholds['mag_thresh']
    else:
        mag_thresh = (30, 100)
    if 'dir_thresh' in thresholds:
        dir_thresh = thresholds['dir_thresh']
    else:
        dir_thresh = (0.7, 1.3)
    if 's_chn_thresh' in thresholds:
        s_chn_thresh = thresholds['s_chn_thresh']
    else:
        s_chn_thresh = (170, 255)
    if 'h_chn_thresh' in thresholds:
        h_chn_thresh = thresholds['h_chn_thresh']
    else:
        h_chn_thresh = (50, 100)
    if 'b_chn_thresh' in thresholds:
        b_chn_thresh = thresholds['b_chn_thresh']
    else:
        b_chn_thresh = (145, 255)
    if 'g_thresh' in thresholds:
        g_thresh = thresholds['g_thresh']
    else:
        g_thresh = 170

    #grad_x = abs_sobel_thresh(img, orient='x', kernel=grad_ksize, thresh=grad_thresh, channel='gray')
    #grad_y = abs_sobel_thresh(img, orient='y', kernel=grad_ksize, thresh=grad_thresh, channel='gray')
    gray_mag_binary = mag_threshold(img, kernel=mag_ksize, thresh=mag_thresh, channel='gray')
    gray_dir_binary = dir_threshold(img, kernel=dir_ksize, thresh=dir_thresh, channel='gray')
    b_mag_binary = mag_threshold(img, kernel=mag_ksize, thresh=mag_thresh, channel='b')
    b_dir_binary = dir_threshold(img, kernel=dir_ksize, thresh=dir_thresh, channel='b')

    s_binary = hls_select(img, thresh=s_chn_thresh, channel='s')
    b_binary = lab_select(img, thresh=b_chn_thresh)
    g_binary = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > g_thresh

    right_lane_combined = np.zeros_like(gray_mag_binary)
    right_lane_combined[((gray_mag_binary == 1) & (gray_dir_binary == 1)) & ((g_binary == 1) | (s_binary == 1))] = 1

    left_lane_combined = np.zeros_like(s_binary)
    left_lane_combined[(b_binary == 1)] = 1

    color_binary = np.dstack((np.zeros_like(left_lane_combined), left_lane_combined, right_lane_combined)) * 255
    color_binary = np.maximum(np.minimum(255, color_binary), 0).astype(np.uint8)

    combined = np.zeros_like(left_lane_combined)
    combined[(left_lane_combined == 1) | (right_lane_combined == 1)] = 1

    return combined, color_binary

def perspective_transform(img, src, dst):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(src)
    dst = np.float32(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv

def scan_find_lane(warped, nwindows=9, margin=100, minpix=50, plot=False):
    h, w = warped.shape[0], warped.shape[1]

    out_img = np.dstack((warped, warped, warped))*255

    histogram = np.sum(warped[h//2:,:], axis=0)
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(6, 8), subplot_kw={'xticks': [], 'yticks': []})
        ax[0].plot(histogram)
        ax[0].set_xlabel('Pixel Position')
        ax[0].set_ylabel('Counts')
        ax[0].set_xticks(np.arange(0, w, 200))
        ax[0].set_yticks(np.arange(0, np.max(histogram), 50))
        ax[0].set_xlim((0, w))
        ax[0].set_ylim((0, np.max(histogram)))

    # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    midpoint = w // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = h // nwindows

    # Identify the x and y positions of all nonzero pixels in the image
    nonzeroy, nonzerox = warped.nonzero()

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = h - (window+1)*window_height
        win_y_high = h - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                (0,255,0), thickness=4) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                (0,255,0), thickness=4)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
             (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return_vars = {}

    ploty = np.linspace(0, h-1, h)
    return_vars['fit_y'] = ploty

    if len(left_lane_inds) < 3:
        return_vars['left_found'] = False
    else:
        return_vars['left_found'] = True
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 

        left_fit = np.polyfit(lefty, leftx, 2)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        return_vars['left_fit_eq'] = left_fit
        return_vars['left_fit_x'] = left_fitx

    if len(right_lane_inds) < 3:
        return_vars['right_found'] = False
    else:
        return_vars['right_found'] = True
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        right_fit = np.polyfit(righty, rightx, 2)

        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return_vars['right_fit_eq'] = right_fit
        return_vars['right_fit_x'] = right_fitx

    if len(left_lane_inds) >= 3 and len(right_lane_inds) >= 3:
        y_eval = h
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7 / (right_fitx[-1] - left_fitx[-1])

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
            / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
            / np.absolute(2*right_fit_cr[0])

        return_vars['left_curverad'] = left_curverad
        return_vars['right_curverad'] = right_curverad
    
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if plot:
        ax[1].imshow(out_img)
        ax[1].plot(return_vars['left_fit_x'], return_vars['fit_y'], c='yellow', lw=2)
        ax[1].plot(return_vars['right_fit_x'], return_vars['fit_y'], c='yellow', lw=2);
    
    return return_vars, out_img

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), 
            max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def mask_boundary(margin, height, img_ref, center, level):
    y_low = int(img_ref.shape[0]-(level+1)*height)
    y_high = int(img_ref.shape[0]-level*height)
    x_low = max(0,int(center-margin))
    x_high = min(int(center+margin),img_ref.shape[1])

    return y_low, y_high, x_low, x_high

def find_window_centroids(image, window_width, window_height, margin):
    height, width = image.shape[0], image.shape[1]

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*height/4):, :width//2], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width//2
    r_sum = np.sum(image[int(3*height/4):, width//2:], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width//2 + width//2 
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,height//window_height):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(height-(level+1)*window_height):
            int(height-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width//2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def conv_find_lane(warped, n_window=9, window_width=50, margin=100):
    window_height = warped.shape[0] // n_window
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    out_img = np.dstack((warped, warped, warped))*255

    nonzeroy, nonzerox = warped.nonzero()

    # If we found any window centers
    if len(window_centroids) > 0:
        left_lane_inds = []
        right_lane_inds = []

        # Go through each level and draw the windows 	
        for level in range(len(window_centroids)):
            # Window_mask is a function to draw window areas
            win_y_low, win_y_high, win_xleft_low, win_xleft_high = \
                mask_boundary(margin, window_height, warped, window_centroids[level][0], level)
            win_y_low, win_y_high, win_xright_low, win_xright_high = \
                mask_boundary(margin, window_height, warped, window_centroids[level][1], level)

             # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                            (0,255,0), thickness=4) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                (0,255,0), thickness=4)

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    y_eval = warped.shape[0]
    ym_per_pix = 26/720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / (right_fitx[-1] - left_fitx[-1])

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
        / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
        / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    curve = {'y': ploty}
    curve['left_x'] = left_fitx
    curve['right_x'] = right_fitx

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, out_img, curve, left_curverad, right_curverad

def drawPoly(img, leftx, lefty, rightx, righty, M = None):
    mix_img = np.zeros_like(img)

    left_pts = np.c_[leftx, lefty].astype(np.int32)
    right_pts = np.c_[rightx, righty].astype(np.int32)
    pts = np.hstack((left_pts, right_pts))
    cv2.polylines(mix_img, [left_pts.reshape((-1,1,2))], isClosed=False, color=(255,0,0), thickness=10)
    cv2.polylines(mix_img, [right_pts.reshape((-1,1,2))], isClosed=False, color=(0,0,255), thickness=10) 
    cv2.fillPoly(mix_img, [pts.reshape((-1,1,2))], (30, 255, 90))

    if M is not None:
        mix_img = cv2.warpPerspective(mix_img, M, mix_img.shape[1::-1], flags=cv2.INTER_LINEAR)

    img = cv2.addWeighted(img, 1, mix_img, 0.5, 0)

    return img

def addText(img, left_curverad, right_curverad, M = None):
    h, w = img.shape[0], img.shape[1]
    mix_img = np.zeros_like(img)
    out_text = 'Radius of Curvature = {:.0f} (m)'.format(np.maximum(left_curverad, right_curverad))
    cv2.putText(mix_img, text = out_text, 
        org = (w//20,h//10), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3, color=(255,255,255))
    if M is None:
        mix_img = cv2.warpPerspective(mix_img, M, mix_img.shape[1::-1], flags=cv2.INTER_LINEAR)

    img = cv2.addWeighted(img, 1, mix_img, 0.7, 0)

    return img

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, sample_window, margin):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_x_fitted = deque([])
        # polynomial coefficients of the last n fits
        self.recent_fit_eq = deque([])
        #average x values of the fitted line over the last n iterations
        self.best_x = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_eq = None  
        #x value of the most recent fit
        self.current_x_fitted = None
        #polynomial coefficients for the most recent fit
        self.current_fit_eq = np.array([False])
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        self.sample_window = sample_window
        self.margin = margin

    def mask(self, img):
        mask_img = np.zeros_like(img)
        if self.best_x is None:
            return mask_img[:,:,2]
        else:
            plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
            left_pts = np.c_[np.array([np.min(i) for i in zip(*self.recent_x_fitted)])
                            - self.margin, plot_y].astype(np.int32)
            right_pts = np.c_[np.array([np.max(i) for i in zip(*self.recent_x_fitted)])
                             + self.margin, plot_y].astype(np.int32)

            mask_img = cv2.fillPoly(mask_img, [np.hstack([left_pts, right_pts]).reshape((-1,1,2))], 
                color=(0,0,255))
            
            binary_mask = (mask_img[:,:,2] == 255)
            return binary_mask
    
    def update(self, detected, x_fitted, fit_eq, rad):
        self.detected = detected

        if detected:
            self.recent_x_fitted.append(x_fitted)
            self.recent_fit_eq.append(fit_eq)

            if len(self.recent_x_fitted) > self.sample_window:
                self.recent_x_fitted.popleft()
                self.recent_fit_eq.popleft()
            
            self.best_x = np.array([np.mean(i) for i in zip(*self.recent_x_fitted)])
            self.best_fit_eq = np.array([np.mean(i) for i in zip(*self.recent_fit_eq)])

            self.current_x_fitted = x_fitted
            self.current_fit_eq = fit_eq
            self.radius_of_curvature = rad

    def sanity_check(self, x_fitted):
        return np.max(np.abs(self.current_x_fitted - x_fitted)) < 100