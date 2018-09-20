# Advanced Lane Finding Project

This project introduces new techniques to find lanes in an image or video. It uses traditional computer vision techniques which require a lot of hand tuning of parameters.

In essence, we use the color/gradient information cleverly to isolate lane lines , use a transform to get a bird's eye view. We then use gradient information to detect lane lines pixels and fit a line through them.

We also deal with real world issues like camera distortion and calculate the curve of the lanes.

This project gave me a sense of how much a lane detection system needs to be tuned for particular road conditions and how much work it is to make it robust.

## How to run ?

1. Install anaconda
2. Install the python packages
	
	```
	conda env create -f environment.yml
	```
3. Run the pipeline
	
	```
	conda activate carnd-term1
	python pipeline.py
	python movie.py
	```


[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration1.jpg "Undistorted"
[image1_1]: ./output_images/plots/undistorted_lane_images/test1.jpg "Undistorted"
[image2]: ./output_images/plots/binary_lane_images/test3.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/plots/transformed_lane_images/test3.jpg
[image5]: ./tmp/lane_lines/straight_lines2.jpg
[image6]: ./output_images/lane_images_with_lane_lines/test2.jpg
[video1]: ./project_video.mp4 "Video"

# Rubric Points

## Camera Calibration

[pipeline.py](pipeline.py) lines 4-82 has functions that perform the distortion.

The idea is use OpenCV `cv2.findChessboardCorners` to find corners on the chessboard images. These corners are then used by another OpenCV function `cv2.calibrateCamera` which gives us a camera matrix and distortion coefficient.

Once we have the camera matrix and distortion coefficient, we can use these to undistort the images.

[pipeline.py](pipeline.py) lines 386-406 shows this in action. An example output image is 

![alt text][image1]



## Pipeline for images

The pipeline processes the images to find lane lines. The processing steps are:

- Camera Distortion Correction
- Use color and gradients to create a thresholded binary image
- Apply perspective transform to get a birds-eye view
- Once we get a processed binary warped image, we find lane pixels and fit a curve through the pixels.
- We then revert the perspective transform and paint the lanes on the images.

We also calculate the radius of curvature of the lanes and position of the car from center of the lane. 


####Distortion correction


The pipeline applies distortion correction using the camera matrix and distortion coefficients from camera caliberation step to all the test images. 

The code is here : [pipeline.py](pipeline.py) lines 406-433

An example of distortion correction applied to one of the test images 

![alt text][image1_1]

The comparison plots are in here : [plots](./output_images/plots/undistorted_lane_images)

The undistorted test images are here : [undistorted test images](./output_images/undistorted_lane_images)


#### Thresholded binary image
After a lot of experimentation with Sobel operators and color spaces. I first converted the image to grayscale as we dont need all 3 color spaces for this step. 

I used the sobel operator to calculate the derivative in the `x` direction. I found that the derivate in x direction worked better than `y` direction. After some experimentation, I found that the `min threshold = 20` and `max threshold = 100` works best for selecting the binary output.

I chose to use the HSV color space as described in the video. After experimentation, I chose to use the saturation channel.

The functions for applying sobel operators and processing color spaces are here: [pipeline.py](pipeline.py) 84-165

These functions are used to get a binary thresholded images from the test images. A sample image is shown below

![alt text][image2]

The code that uses these functions is here: [pipeline.py](pipeline.py) lines 446-468

The comparison plots are in here : [plots](./output_images/plots/binary_lane_images)

The undistorted test images are here : [binary test images](./output_images/binary_lane_images)

#### Perspective transform

The perspective transform then transforms the binary file to a bird's eye view that makes the lanes much straighter. It also makes it easier to fit a polynomial through the lane pixels. 

The functions that calculates the perspective transform matrices and transforms the images based on these tranform matrices is here: [pipeline.py](pipeline.py) lines 176-201

The `perspective_transform_matrices()` function takes as inputs an image (`img`) and source (`src`) points.  I chose the hardcode the source and destination points in the following manner:

```python
bottom_left, top_left, top_right, bottom_right = (
            [100, img_y - 50],
            [550, 470],
            [800, 470],
            [img_x - 100, img_y - 50])
```
```
dst = np.float32([
        [offset, img_y - offset],
        [offset, offset],
        [img_x - offset, offset],
        [img_x - offset, img_y - offset]

    ])
```

The code that applies the perspective transform to binary thresholded test images is here: [pipeline.py](pipeline.py) lines 470-509. 

An example image along with transformed output is shown below

![alt text][image4]

The comparision plots are in here : [plots](./output_images/plots/transformed_lane_images)

The undistorted test images are here : [transformed test images](./output_images/transformed_lane_images)

#### Identifying lane-line pixels and fitting a polynomial curve

The function that selects the lane pixels and fits a polynomial are here: [pipeline.py](pipeline.py) lines 204-294. 

The function `find_lanes_sliding_window` does a full sliding window search on the image. `find_lanes_non_sliding_window` function optimizes the search by searching in a margin around the previous line position.

The code that selects the lane pixels and fits a polynomial for the transformed images is here: [pipeline.py](pipeline.py) lines 470-509. 

An example of this transformation is shown below.
![alt text][image5]

#### Radius of curvature of the lane and position of the vehicle with respect to center.

The functions that calculate the radius of curvature and position of vehicle is here: [pipeline.py](pipeline.py) lines 297-321. 

#### Plotting lane lines on the images

The function that draws the lane on the image is here : [pipeline.py](pipeline.py) lines 324-346

The code that processes all image files is [pipeline.py](pipeline.py) lines 516-561

One of the processed test images is shown below.

![alt text][image6]

The test images with lane lines drawn are here : [images with lane lines](./output_images/lane_images_with_lane_lines)



## Pipeline (video)

For the video, I used the sliding window search for the first frame and non sliding window (prior search) for all subsequent frames. My first attempt was naively based on applying these functions to the video but the output was choppy. 

Based on a suggestion from one of fellow students, I used weighted averaging of lane pixels over the last 10 windows. It made the video processing smooth and accurate.

Here's a [link to the video result](./output_videos/project_video.mp4)

The code is here : [video.py](video.py)


## Discussion

This assignment required a lot of hand tuning of parameters. I would like to research if deep learning techniques are better than hand tuning the parameters. The techniques used are not very robust to changes in the color or lighthing condtions. Also, if the lane curves steeply, I suspect the pipeline would fail. I would do more research into fixing this.
