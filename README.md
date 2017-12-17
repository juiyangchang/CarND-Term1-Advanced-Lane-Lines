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
[image6]: ./examples/example_output.jpg "Output"
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
in section **Camera Calibration** in the Ipython notebook.

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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`