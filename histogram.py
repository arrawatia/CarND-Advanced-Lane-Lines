import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def read_image(file):
    return mpimg.imread(file)


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


lane_images = glob.glob('output_images/transformed_lane_images/*.jpg')

output_path_plots = "tmp/histogram_lane_lines"
if not os.path.exists(output_path_plots):
    os.makedirs(output_path_plots)

for lane_image in lane_images:
    # `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
    img = cv2.cvtColor(read_image(lane_image), cv2.COLOR_BGR2GRAY)

    # %%
    img_x, img_y = img.shape[1], img.shape[0]

    # Create histogram of image binary activations
    histogram = hist(img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.plot(histogram)
    # ax2.set_title('Thresholded Gradient', fontsize=50)
    ax2.imshow(histogram)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))
