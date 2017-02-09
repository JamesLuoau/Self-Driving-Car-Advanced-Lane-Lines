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

[image1]: ./output_images/calibration1_undist.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[test1]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image_area_peak]: ./other_images/histogram_area_instead_of_peak_point.png "histogram_area_instead_of_peak_point"
[image_sobel_combined]: ./output_images/image_combined_4.jpg "sobel find combined"
[perspective]: ./output_images/perspective/test6.jpg.png "perspective"

##Before Start
The whole code structure has been designed for fast debug
run **main.py**, it will create target video, there is a image index in very frame, if you see any frame not right
you can just pick up the images from **video_images** folder with same index file name, put this file into 
**test_images**, run whole unit test in **/test/sobel_test.py**, it will create images for every step so that you can 
do a detailed check why it's failing.


###Camera Calibration

The code for this step is contained in camera_calibrate.py.


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners 
in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the 
object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, 
and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners 
in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane 
with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion 
coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

I choose calibration1.jpg as a test image to un-distortion, also saves camera matrix and distortion into a file 
called "camera_calibration_pickle.p" for later usage.

```python
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
camera_matrix, distortion = calibrateCamera(img_size)
save_camera_calibration(camera_matrix, distortion)
undistort(camera_matrix, distortion, img, './output_images/calibration1_undist.png')
```
![alt text][image1]

###Pipeline (single images)

####1. Binary image generation with color and gradient thresholds 
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `thresholding.py`).  
Here's an example of my output for this step.
To generate below images, please run `test_threshold` method under `pipe_line_test.py` 
![alt text][test1]
![alt text][image_sobel_combined]

```python
def pipeline(img):
    img = np.copy(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)

    combined = combine_with_or(
        abs_sobel_thresh(gray_image, orient='x', sobel_kernel=25, thresh=(50, 150)),
        combine_with_or(
            *bgr_channel_threshold(img, b_thresh=(220, 255), g_thresh=(220, 255), r_thresh=(220, 255))
        ),
        combine_with_and(
            hls_channel_threshold(hls_image, s_thresh=(170, 255))[2],
            abs_sobel_thresh(gray_image, orient='x', sobel_kernel=5, thresh=(10, 100))
        )
    )
    return combined

```

####2. Perspective transform
All perspective transform method are located in `perspective_transform.py`
`def perspective_transform(img, perspective_transform_matrix)` will transform given image with transform matrix
`def inversion_perspective_transform(img, invent_perspective_transform_matrix)` will inversion that process 

![alt text][perspective]

```python
img = cv2.imread(fname)
matrix, invent_matrix = calculate_transform_matrices(img.shape[1], img.shape[0])
perspective_img = perspective_transform(img, matrix)
save_image(perspective_img, '../output_images/perspective/{}.png'.format(fname))
```

####3. Fit positions with a polynomial from lane-line pixels

#####3.1 Cached Search Base Position
once we know roughly where are two lanes, we can cache it and next frame could search in the similar area. 
the code in main.py
```python
@staticmethod
def _line_search_base_position(histogram,
                               last_know_leftx_base=None, last_know_rightx_base=None, peak_detect_offset=80):
    if last_know_leftx_base is None or last_know_rightx_base is None:
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        left_start, left_end, right_start, right_end = LaneFinder._range(
            len(histogram), last_know_leftx_base, last_know_rightx_base, peak_detect_offset)
        leftx_base = np.argmax(histogram[left_start:left_end]) + left_start
        rightx_base = np.argmax(histogram[right_start:right_end]) + right_start

    return leftx_base, rightx_base
```
the test case will explain how this function works.
in test case 1, if there is no last know line position, the search will start from middle point of histogram, which is 
the index 1 and 4.

In test case 2, it will search across last know point +- peak_detect_offset
```python
def test_line_search_base_position_should_find_middle_point_if_no_last_knowledge(self):
        histogram = np.array([1, 2, 1, 3, 4, 3])
        left, right = LaneFinder._line_search_base_position(histogram, None, None)
        self.assertEqual(left, 1)
        self.assertEqual(right, 4)

def test_line_search_base_position_should_find_peak_point_near_last_know_position(self):
    histogram = np.array([1, 4, 1, 2, 1, 3, 4, 3, 5, 3])
    left, right = LaneFinder._line_search_base_position(histogram, None, None)
    self.assertEqual(left, 1)
    self.assertEqual(right, 8)
    left, right = LaneFinder._line_search_base_position(
        histogram, last_know_leftx_base=4, last_know_rightx_base=6, peak_detect_offset=1)
    self.assertEqual(left, 5)
    self.assertEqual(right, 6)
    left, right = LaneFinder._line_search_base_position(
        histogram, last_know_leftx_base=4, last_know_rightx_base=9, peak_detect_offset=2)
    self.assertEqual(left, 6)
    self.assertEqual(right, 8)
```
#####3.2 Abnormal detection and auto-correction


![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Historgram maybe shouldn't look at the highest point, should look at the highest area. As show in below 
![alt text][image_area_peak]

####2. The implement here are very sensitive to lights
