import cv2
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use('ggplot')


def read_image(file):
    return mpimg.imread(file)


# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # cv2.imshow('img', undist)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    warped = None
    M = None
    if ret == True:
        # If we found corners, draw them! (just for fun)
        # cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


# %%

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# vertices = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
def draw_region(img, vertices, line_color=(0, 255, 0)):
    vertices = vertices.reshape((-1, 1, 2))
    return cv2.polylines(img, [vertices], True, line_color, 3)


# For source points I'm grabbing the outer four detected corners
# src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
def perspective_xform(img, offset, src):
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
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, transform_matrix, (img_x, img_y))

    return warped, transform_matrix


import glob
import os

lane_images = glob.glob('output_images/undistorted_lane_images/*.jpg')

output_path_plots = "tmp/binary_lane_images"
if not os.path.exists(output_path_plots):
    os.makedirs(output_path_plots)

for lane_image in lane_images:
    img = read_image(lane_image)

    # %%
    img_x, img_y = img.shape[1], img.shape[0]

    # bottom_left, top_left, top_right, bottom_right = ([0, img_y], [570, 470], [820, 470], [img_x, img_y])
    # bottom_left, top_left, top_right, bottom_right = ([0, img_y-50], [570, 470], [820, 470], [img_x, img_y-50])
    # bottom_left, top_left, top_right, bottom_right = ([0, img_y-50], [570, 470], [820, 470], [img_x, img_y-50])

    # Crop hood (crop bottom 50 pixels)
    y_bottom = img_y - 50

    # The lane converges around 60% from the top of the images
    y_top = 0.6 * y_bottom

    # The top of the trapeziod (10% of the width of the image and centered)
    center = img_x / 2

    top_width = img_x * 0.095

    top_right = [center + top_width, y_top]
    top_left = [center - top_width, y_top]

    # The bottom of the trapeziod (40% of the width of the image and centered)
    bottom_width = img_x * 0.35

    bottom_right = [center + bottom_width, y_bottom]
    bottom_left = [center - bottom_width, y_bottom]

    bottom_left, top_left, top_right, bottom_right = (
        [100, img_y - 50],
        [550, 470],
        [800, 470],
        [img_x - 100, img_y - 50])

    vertices = [bottom_left, top_left, top_right, bottom_right]

    print("image=%s, center=%s, vertices=%s" % (img.shape, center, vertices))

    r = draw_region(img, np.array(vertices, np.int32))

    warped, transform_matrix = perspective_xform(img, 0, np.array(vertices, np.float32))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped)
    # ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))
