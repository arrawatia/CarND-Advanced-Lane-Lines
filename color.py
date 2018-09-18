# %%
import cv2
import matplotlib
import matplotlib.image as mpimg
# matplotlib.use('Qt5Agg')
# This should be done before `import matplotlib.pyplot`
# 'Qt4Agg' for PyQt4 or PySide, 'Qt5Agg' for PyQt5
import matplotlib.pyplot as plt
import numpy as np

matplotlib.style.use('ggplot')


# %%


# %%
def read_image(file):
    return mpimg.imread(file)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


# %%
img = read_image('curved-lane.jpg')
plt.imshow(img, cmap='BrBG')

gray = grayscale(img)
plt.imshow(gray, cmap='gray')
plt.imshow(abs_sobel_thresh(gray, mag_thresh=(20, 100)), cmap='gray')

image = read_image('signs_vehicles_xygrad.png')
plt.imshow(image)
grad_binary = abs_sobel_thresh(grayscale(image), orient='x', mag_thresh=(20, 100))
plt.imshow(grad_binary, cmap='gray')
grad_binary = abs_sobel_thresh(grayscale(image), orient='y', mag_thresh=(20, 100))
plt.imshow(grad_binary, cmap='gray')

plt.imshow(mag_thresh(grayscale(image), sobel_kernel=3, mag_thresh=(30, 100)), cmap='gray')

plt.imshow(dir_threshold(grayscale(image), sobel_kernel=15, thresh=(0.7, 1.3)), cmap='gray')

# %%

image = read_image('signs_vehicles_xygrad.png')

ksize = 3  # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gray = grayscale(image)
gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, mag_thresh=(20, 100))
grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, mag_thresh=(20, 100))
mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plt.imshow(combined, cmap='gray')

# %%

image = read_image('curved-lane.jpg')
plt.imshow(image)
plt.imshow(hls_select(image, thresh=(20, 80)), cmap='gray')
# %%
image = read_image('curved-lane.jpg')
sxbinary = abs_sobel_thresh(grayscale(image), orient='x', sobel_kernel=ksize, mag_thresh=(20, 100))
s_binary = hls_select(image, thresh=(170, 255))
color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
plt.imshow(color_binary)

combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
plt.imshow(combined_binary, cmap='gray')

# %%
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
