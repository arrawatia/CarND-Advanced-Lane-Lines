import glob
from collections import deque

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
import pipeline

num_bins = 10
left_lane_lines = deque(maxlen=num_bins)
right_lane_lines = deque(maxlen=num_bins)
left_radius = deque(maxlen=num_bins)
right_radius = deque(maxlen=num_bins)
weights = np.arange(1, num_bins + 1) / num_bins

images = glob.glob('camera_cal/calibration*.jpg')
nx = 9
ny = 6
test_image = pipeline.read_image('camera_cal/calibration1.jpg')
img_size = (test_image.shape[1], test_image.shape[0])

return_value, camera_matrix, dist_coeffs = pipeline.calibrate(images, img_size, nx, ny)


def weighted_average(points, weights):
    return np.average(points, 0, weights[-len(points):])


def process_image(image):
    undist_lane_image = pipeline.undistort(image, camera_matrix, dist_coeffs)
    binary_image = pipeline.thresholded_image(undist_lane_image, 3)

    img_x, img_y = image.shape[1], image.shape[0]

    bottom_left, top_left, top_right, bottom_right = (
        [100, img_y - 50],
        [530, 470],
        [820, 470],
        [img_x - 100, img_y - 50])

    vertices = [bottom_left, top_left, top_right, bottom_right]
    transform_matrix, inverse_transform_matrix = pipeline.perspective_transform_matrices(image, 0,
                                                                                         np.array(vertices, np.float32))
    warped = pipeline.warp_image(binary_image, transform_matrix)

    if len(left_lane_lines) == 0:
        leftx, lefty, rightx, righty = pipeline.find_lanes_sliding_window(warped)
    else:
        leftx, lefty, rightx, righty = pipeline.find_lanes_non_sliding_window(warped,
                                                                              weighted_average(left_lane_lines,
                                                                                               weights),
                                                                              weighted_average(right_lane_lines,
                                                                                               weights))

    left_fit, right_fit = pipeline.fit(leftx, lefty, rightx, righty)
    left_lane_lines.append(left_fit)
    right_lane_lines.append(right_fit)

    left_curverad, right_curverad = pipeline.radius_of_curvature(warped, left_fit, right_fit, leftx, lefty,
                                                                 rightx,
                                                                 righty)
    left_radius.append(left_curverad)
    right_radius.append(right_curverad)

    position = pipeline.offset(image, left_fit, right_fit)
    result = pipeline.draw_lane(image, warped, weighted_average(left_lane_lines, weights),
                                weighted_average(right_lane_lines, weights), inverse_transform_matrix)
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

    return result


output_path = "output_videos"

if not os.path.exists(output_path):
    os.makedirs(output_path)
output = 'output_videos/project_video.mp4'
clip = VideoFileClip("project_video.mp4")
white_clip = clip.fl_image(process_image)
white_clip.write_videofile(output, audio=False)
