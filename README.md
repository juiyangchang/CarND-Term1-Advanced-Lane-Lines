# **Advanced Lane Finding Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  ![Language](https://img.shields.io/badge/language-Python-green.svg)

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
[image1]: ./output_images/camera_calibration.png "Undistorted"
[image2]: ./output_images/pipeline_undistort.png "Undistorted Image on Road"
[image3]: ./output_images/thresholding.png "Thresholded Binary Image"
[image4]: ./output_images/perspective_transform.png "Warped Image"
[image5]: ./output_images/polynomial_fit.png "Fitted Polynomial"
[image6]: ./output_images/pipeline_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The whole pipeline for camera calibration and undistortion can be found
in section **Camera Calibration** in the [Ipython notebook](Advanced%20Lane%20Finding%20Project.ipynb).

I started by creating the object points `objp` in the chessboard.
Then the chessboard corner image points are found from all but the first image
in the `camera_cal` folder.  Each images is first converted from RGB to Gray scale,
then the chessboard corner image points are found using `cv2.findChessboardCorners`.
The object points and image points of all images are collected in two python lists
and fed into the function `calibrate`, defined in `ImageProcessing/process.py`.
This function returns the camera matrix and distortion coefficients.
The camera matrix and the distortion coefficients are then passed to the function `undist` to undistort the image.
Below is an example of undistorted image, in which the original image was not used in calibration process.
![Undistorted Image][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Below is an example of distortion-corrected on-road image.
![Undistorted Image on Road][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The procedures for converting the image to thresholded binary image is implemented in the function `thresholding()` in `ImageProcessing/process.py`. 

The yellow lane lines are identified by thresholding B channels of the LAB color space. I took reference from a [former student of this program](https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines). In the middle panel in the figure below, we can see the thresholded B channel (geen lines) successfully identifies the yellow lane.

The white lane lines are identified with thresholded magnitude of the graident, direction of graident, thresholded S channel of HLS color space and thresholded gray scale image. 
The logical rule for the white lane line detection procedure look like this:
```python
white_lane_combined[((mag_binary == 1) & (dir_binary == 1)) & ((gray_binary == 1) | (s_binary == 1))] = 1
```
In the middle panel in the figure below the identified white lanes are shown in blue lines. 
In the rightmost panel I show the binary image result.
![Thresholded Binary Image][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform procedure is imlemented in the function `perspective_transform()` in `ImageProcessing/process.py`.  It calls `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to get the perspective transform matrix and make perspective transform.  It also returns an inverse transform matrix.

I identified the source points and destination points with the following coordinates:
```python
h, w = image.shape[:2]
SRC = np.array([[1.7 * w // 10, h], [8.7 * w // 10, h], [5.75 * w // 10, h - h // 3], 
                    [4.33 * w // 10, h - h // 3]])
DST = np.array([[2 * w // 10, h], [8 * w // 10, h], [8 * w // 10, h // 2], 
                    [2 * w // 10, h // 2]])
```
Below I show the thresholded image and perspective transformed image side by side in the middle and right panels.  The red lines indicates the source and destination points.  The figure shows that the perspective transform successfully transform the binary thresholded image to top-down view and the identified lanes appear to be parallel in the figure.
![Warped Image][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The procedure for fitting a quadratic function to the pixels in the binary thresholded image is implemented in the 
function `scan_find_lane()` defined in `ImageProcessing/process.py`.  

I follow the procedure described in the lecture. To start, I computed the vertical counts of onset pixels from 
vertical mid point to bottom of the binary image in each horizontal pixel location.
The counts are shown in the top panel in the figure below.  The pixel points with the largest counts on the left and
right halfs are then used intial center points of the left and right windows.  The center points can be seen as shown
in the center point x coordinates of the bottom green rectangles in the bottom panel.

The image is vertically divided into nine windows. We start scanning from the center points we mentioned before from
the bottom windows. The windows are of width 240.  The onset pixels in the left window is included in the list of points that will be
used for fitting the left quadratic function, we also keep another list for the right lane.  We also count the number
onset pixels in the windows. If the number exceeds 50, we will move the center point of the next window to the
mean of x coordinates the onset pixels in the window.  We then repeat the procedure for the second vertical window
from the bottom. These procedures are repeated until the ninth window.  In the figure, the green rectangles
depict the windows. The red pixels are the pixels included in the list of points that will be fitted to find the left
curve, while the blue pixels will be used to find the right curve.

Finally, to find the quadratic functions for the cuves, we collect the y and x coordinates of the pixels into
two numpy arrays `y` and `x`.  We then fit `x` as a quadratic function of `y` with `np.polyfit(y, x, deg=2)`. The fitted quadratic functions are plotted in yellow in the bottom panel of the figure.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane is calcuated in `scan_find_lane()` in `ImageProcessing/process.py`.  

The main idea is that we can use the coefficients of the fitted quadratic function to compute the raidus of curvature, followying the equation noted in this [page](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).  Here instead of using number of pixels as the distance unit, we need to use real world distance in meters.  I followed the class note to set the meters per pixel in the vertical direction to `30/720` meters per pixel.  As for the horizontal direction, I used the distance between the fitted lines at the bottom of the picture `lx` and `rx` and set the conversion to `3.7 / (rx - lx)` meters per distance.  Once we converted the the pixel coordinates be of unit meters, we can fit the
quadratic polynomials again and use their coefficients to calculate the radius of curvature. 

As for the center position of the vehicle, I used the distance between  the x axis mid point of the picture 
and the mid point of the two fitted curves
at the bottom of the picture to compute the relative location of the car.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The whole pipeline is collected in the `pipeline()`
function defined in `ImageProcessing/pipeline.py`.  Below is the output image of my pipeline.

![alt text][image6]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Below is a link to my project video results:
[![Project Video](http://img.youtube.com/vi/opuoJeqBzUA/0.jpg)](https://youtu.be/opuoJeqBzUA "Project Video")
you can also find it in this [repo](output_vides/project_video.mp4).


Below is a link to my challenge video results:
[![Challenge Video](http://img.youtube.com/vi/kRrMwEymS-0/0.jpg)](https://www.youtu.be/kRrMwEymS-0 "Challenge Vide")
you can also find it in this [repo](output_vides/challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some shadows on the road can be of gradient with similar strength with the lane line and as saturated in S channel. Sometimes shadows or strong sun lit can completely block the road. Another challenge is that
roads can be really curvy as in the harder challenge video, which make it difficult for the scanning procedure to recenter the scanning window.  

In my pipeline, switching to B channel of the LAB color space helped make the procedure more robust.  I also used lane lines fitted in the previous five video frame to create a mask to fileter out distant pixels from the lane line.
If the procedure failes for the current frame, I will just draw the lines detected in the previous video frame.

However my pipeline is not robust enough to find lane line in the harder challenge video. I think using image contrast strengthening procedures like histogram equalization and Kalman filtering or similar filtering procedures might be helpeful to find the lane lines. We can also use locations of other cars/motocycles to make
an educated guess of the lane lines.  If other sensor reads is available, we can fushion those into our pipeline to make a better detection.
