import cv2
import numpy as np
from matplotlib import pyplot as plt

image_prev = cv2.imread('highway_2.png')
image_next = cv2.imread('highway_1.png')

prev_gray = cv2.cvtColor(image_prev, cv2.COLOR_BGR2GRAY)
next_gray = cv2.cvtColor(image_next, cv2.COLOR_BGR2GRAY)

height, width = prev_gray.shape

max_iterations = 30
resolution_levels = 0

# Parameter settings for optical flow
params = dict(
    winSize=(25, 25),
    maxLevel=resolution_levels, # Amount of resolution levels for coarse-to-fine estimation
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 0.01) # Iteration count in iterative refinement
)


# Find optimal corners to track using Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(prev_gray, 100, 0.4, 5)
# Or using ORB...
# orb = cv2.ORB.create()
# keypoints, descriptors = orb.detectAndCompute(prev_gray, None)
# corners = np.array([keypoint.pt for keypoint in keypoints]).reshape(-1, 1, 2).astype(np.float32)

# Calculate the optical flow using Lucas-Kanade with coarse-to-fine estimation
new_pos, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, corners, None, **params)


tracked_features = image_next.copy()

# Selecting the successfully tracked keypoints
new_positions = np.int64(new_pos[status==1])
old_positions = np.int64(corners[status==1])

# Draw lines for the translation of the features between the two images
for points_pair in zip(new_positions, old_positions):
    cv2.line(tracked_features, points_pair[0], points_pair[1], (255,0,0), 2)


# Selects a sparse set of points in the image to calculate the optical flow over the image
step_size = 100
x, y = np.meshgrid(np.arange(height), np.arange(width))
points = np.column_stack([y.ravel(), x.ravel()])
points = points[step_size // 2::step_size]
# Stacked points like: [(0,0)], [(0,1)]
points = points.reshape(-1, 1, 2).astype(np.float32)

# Computes the optical flow for the set of points
new_pos, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, points, None, **params)

new_positions = np.int64(new_pos[status == 1])
old_positions = np.int64(points[status == 1])

optical_flow = image_next.copy()

# Draw arrows to visualize the direction and magnitude of the optical flow
for points_pair in zip(old_positions, new_positions):
    cv2.arrowedLine(optical_flow, points_pair[0], points_pair[1], (255, 0, 0), thickness=1, tipLength=0.3)

plt.figure()
plt.imshow(tracked_features)
plt.figure()
plt.imshow(optical_flow)
plt.title(f'Lucas Kanade with {resolution_levels+1} resolution levels and {max_iterations} iterations')
plt.show()