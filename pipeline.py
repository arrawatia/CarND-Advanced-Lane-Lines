# %%
'''
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## First, I'll compute the camera calibration using chessboard images
'''

import glob
import os

import cv2
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use('ggplot')


def read_image(file):
    return mpimg.imread(file)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Get the undistorted grid of points. This is used as reference by the calibrate camera function to
# create a calibration matrix from the detected corners on the chessboard image.
def un_distorted_grid(nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    return objp


def detect_corners(image, nx, ny):
    img = read_image(image)
    gray = grayscale(img)

    # Find the chessboard corners
    corners_found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return corners_found, corners


def calibrate(images, img_size, nx, ny):
    # corners = detected 2d points in image plane.
    all_corners = []
    for file in images:

        # Find the chessboard corners
        corners_found, corners = detect_corners(file, nx, ny)

        # If found, add object points, image points
        if corners_found is True:
            all_corners.append(corners)

    # Get reference points (3d points in real world space) for each corner
    reference_points = [un_distorted_grid(nx, ny) for _ in all_corners]

    # Do camera calibration given object points and image points
    return_value, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(reference_points, all_corners, img_size, None,
                                                                         None)

    return return_value, camera_matrix, dist_coeffs


def undistort(image, camera_matrix, dist_coeffs):
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)

    # Scale the image to 8 bits (0 to 255). Makes sure the thresholds are correct
    # for all image sizes
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    # Pixels are 0 or 1 based on the strength of the x gradient
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sxbinary


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def thresholded_image(image, ksize):
    sxbinary = abs_sobel_thresh(grayscale(image), orient='x', sobel_kernel=ksize, mag_thresh=(20, 100))
    s_binary = hls_select(image, thresh=(170, 255))

    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


# vertices = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
def draw_region(img, vertices, line_color=(0, 255, 0)):
    vertices = vertices.reshape((-1, 1, 2))
    return cv2.polylines(img, [vertices], True, line_color, 3)


# For source points I'm grabbing the outer four detected corners
# src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
def perspective_transform_matrices(img, offset, src):
    # Grab the image shape
    img_x, img_y = img.shape[1], img.shape[0]

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = np.float32([
        [offset, img_y - offset],
        [offset, offset],
        [img_x - offset, offset],
        [img_x - offset, img_y - offset]

    ])
    # Given src and dst points, calculate the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst, src)

    return transform_matrix, inverse_transform_matrix


def warp_image(img, transform_matrix):
    img_x, img_y = img.shape[1], img.shape[0]
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, transform_matrix, (img_x, img_y))
    return warped


# HYPERPARAMETERS
# nwindows = Choose the number of sliding windows
# margin = Set the width of the windows +/- margin
# minpix = Set minimum number of pixels found to recenter window
def find_lanes_sliding_window(binary_warped, nwindows=9, margin=100, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def find_lanes_non_sliding_window(binary_warped, left_fit, right_fit, margin=100):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    return leftx, lefty, rightx, righty


# ym_per_pix meters per pixel in y dimension
# xm_per_pix meters per pixel in x dimension
def radius_of_curvature(binary_warped, left_fit, right_fit, leftx, lefty, rightx, righty, ym_per_pix=30 / 720,
                        xm_per_pix=3.7 / 700):
    # the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    # print("pixels = ", left_curverad, right_curverad)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    # print("meters = ", left_curverad, right_curverad)
    return left_curverad, right_curverad


def draw_lane(source_image, warped_binary, left_fit, right_fit, transform_matrix):
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, transform_matrix, (source_image.shape[1], source_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(source_image, 1, newwarp, 0.3, 0)
    return result


def fit(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def plot(binary_warped, left_fit, right_fit):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def offset(image, left_fit, right_fit, xm_per_pix=3.7 / 700):
    midpoint = np.int(image.shape[1] / 2)

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix
    return offset


def main():
    images = glob.glob('camera_cal/calibration*.jpg')
    nx = 9
    ny = 6
    test_image = read_image('camera_cal/calibration1.jpg')
    img_size = (test_image.shape[1], test_image.shape[0])

    # 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    return_value, camera_matrix, dist_coeffs = calibrate(images, img_size, nx, ny)

    # Test undistortion on an image
    undist_test_image = undistort(test_image, camera_matrix, dist_coeffs)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(test_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist_test_image, cmap='gray')
    ax2.set_title('Undistorted', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.savefig("output_images/undistorted_calibration1.jpg")

    # * Apply a distortion correction to raw images.
    lane_images = glob.glob('test_images/*.jpg')

    output_path = "output_images/undistorted_lane_images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path_plots = "output_images/plots/undistorted_lane_images"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for lane_image in lane_images:
        image = read_image(lane_image)
        undist_lane_image = undistort(image, camera_matrix, dist_coeffs)
        plt.imsave("%s/%s" % (output_path, os.path.basename(lane_image)), undist_lane_image)

        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist_lane_image, cmap='gray')
        ax2.set_title('Undistorted', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))

    # * Use color transforms, gradients, etc., to create a thresholded binary image.
    lane_images = glob.glob('output_images/undistorted_lane_images/*.jpg')

    output_path = "output_images/binary_lane_images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path_plots = "output_images/plots/binary_lane_images"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    for lane_image in lane_images:
        image = read_image(lane_image)
        sxbinary = abs_sobel_thresh(grayscale(image), orient='x', sobel_kernel=ksize, mag_thresh=(20, 100))
        s_binary = hls_select(image, thresh=(170, 255))

        # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        plt.imsave("%s/%s" % (output_path, os.path.basename(lane_image)), combined_binary, cmap='gray')

        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(combined_binary, cmap='gray')
        ax2.set_title('Thresholded Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))

    # * Apply a perspective transform to rectify binary image ("birds-eye view").
    lane_images = glob.glob('output_images/binary_lane_images/*.jpg')

    output_path = "output_images/transformed_lane_images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path_plots = "output_images/plots/transformed_lane_images"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for lane_image in lane_images:
        img = read_image(lane_image)

        img_x, img_y = img.shape[1], img.shape[0]

        bottom_left, top_left, top_right, bottom_right = (
            [100, img_y - 50],
            [550, 470],
            [800, 470],
            [img_x - 100, img_y - 50])

        vertices = [bottom_left, top_left, top_right, bottom_right]

        # r = draw_region(img, np.array(vertices, np.int32))

        transform_matrix, inverse_transform_matrix = perspective_transform_matrices(img, 0,
                                                                                    np.array(vertices, np.float32))
        warped = warp_image(img, transform_matrix)
        plt.imsave("%s/%s" % (output_path, os.path.basename(lane_image)), warped)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        # ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped)
        # ax2.set_title('Thresholded Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))

    # * Detect lane pixels and fit to find the lane boundary.
    # * Determine the curvature of the lane and vehicle position with respect to center.
    # * Warp the detected lane boundaries back onto the original image.
    # * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    source_lane_images = glob.glob('test_images/*.jpg')
    warped_lane_images = glob.glob('output_images/transformed_lane_images/*.jpg')

    output_path = "output_images/lane_images_with_lane_lines"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path_plots = "output_images/plots/lane_images_with_histogram"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for idx in range(len(lane_images)):
        print(idx, source_lane_images[idx])
        source_image = read_image(source_lane_images[idx])

        warped_lane = cv2.cvtColor(read_image(warped_lane_images[idx]), cv2.COLOR_BGR2GRAY)

        leftx, lefty, rightx, righty = find_lanes_sliding_window(warped_lane)
        left_fit, right_fit = fit(leftx, lefty, rightx, righty)

        # plot(img, left_fit_ns, right_fit_ns)

        left_curverad, right_curverad = radius_of_curvature(warped_lane, left_fit, right_fit, leftx, lefty,
                                                            rightx,
                                                            righty)

        # Given src and dst points, calculate the perspective transform matrix

        position = offset(source_image, left_fit, right_fit)
        result = draw_lane(source_image, warped_lane, left_fit, right_fit, inverse_transform_matrix)
        caption = "Radius of curvature (left = %.2f km, right = %.2f km) Position = %.2f m" % (
        left_curverad / 1000, right_curverad / 1000, position)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_color = (255, 255, 255)
        line_type = 2

        cv2.putText(result, caption,
                    (50, 50),
                    font,
                    font_scale,
                    font_color,
                    line_type)

        plt.imsave("%s/%s" % (output_path, os.path.basename(source_lane_images[idx])), result)

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(source_img)
        # ax2.imshow(result)


if __name__ == '__main__':
    main()
