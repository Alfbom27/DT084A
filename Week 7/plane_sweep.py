import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.png')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

height, width = img1_gray.shape


# Camera calibrations for "chess 2" from Middlesbury web page
B = 124.86
f = 1758.23

# Depth range
depth_range = np.arange(800, 2400, 20)

# Window-size for block matching
block_size = 5
window = np.ones((block_size, block_size), dtype=np.float32)

error_images = []

# For each depth plane
for depth in depth_range:
    disparity = f * B / depth
    disparity_m = np.array([[1, 0, -disparity], [0, 1, 0], [0, 0, 1]])
    # Translation with the disparity
    T = disparity_m
    # Warped to camera 1's view
    warped = cv2.warpPerspective(img1_gray, T, (width, height))

    # Image blurring
    #warped = cv2.GaussianBlur(warped, (5, 5), 1)
    #img2_blur = cv2.GaussianBlur(img2_gray, (5, 5), 1)

    # Cost function, SSD using either a regular box filter or a Gaussian kernel
    squared_diff = (warped - img2_gray) ** 2
    # ssd = cv2.filter2D(squared_diff, -1, window, borderType=cv2.BORDER_REFLECT)
    ssd_g = cv2.GaussianBlur(squared_diff, (15,15), 3.0)
    error_images.append(ssd_g)


# Stack the error images and convert to confidence
cost_volume = np.stack(error_images, axis=2)
confidence = 1 / (cost_volume + 1e-6)

# Aggregate spatially for each depth level using a Gaussian kernel
for d in range(confidence.shape[2]):
    confidence[:, :, d] = cv2.GaussianBlur(confidence[:, :, d], (15, 15), 1)

# Select the best depth at each pixel value
indices = np.argmax(confidence, axis=2)
# Assign depth to each pixel
disparity_map = depth_range[indices]
depth_map = (f * B) / (disparity_map + 1e-6)

# Normalize and convert to uint8 for visualization
depth_map = (depth_map - depth_range.min()) / (depth_range.max() - depth_range.min()) * 255
depth_map = depth_map.astype(np.uint8)


cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(depth_map, cmap='jet')
plt.colorbar(label='Depth')
plt.title('Depth Map')
plt.show()