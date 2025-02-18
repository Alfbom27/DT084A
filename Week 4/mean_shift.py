import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

img = cv2.imread("../Week 2/big_ben2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


height, width, _ = img.shape

# Blur and compute magnitudes of the gradients
#img = cv2.GaussianBlur(img, (3, 3), 0)
gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
gradient_mag_b = np.sqrt(gradient_x[...,0]**2 + gradient_y[...,0]**2)
gradient_mag_g = np.sqrt(gradient_x[...,1]**2 + gradient_y[...,1]**2)
gradient_mag_r = np.sqrt(gradient_x[...,2]**2 + gradient_y[...,2]**2)

x, y = np.meshgrid(np.arange(width), np.arange(height))

# Create the multidimensional space

# Using spatial positions and color values
feature_space = np.stack([x.ravel(), y.ravel(), img[..., 0].ravel(), img[..., 1].ravel(), img[..., 2].ravel()], axis=-1)

# Using spatial positions and color gradients
# feature_space = np.stack([x.ravel(), y.ravel(), gradient_mag_b.ravel(), gradient_mag_g.ravel(), gradient_mag_r.ravel()], axis=-1)

# Estimates optimal bandwidth of the kernel. Higher or lower quantile for wider/smaller bandwidth
bandwidth = estimate_bandwidth(feature_space, quantile=0.02, n_samples=500)
print(f"Bandwidth: {bandwidth}")

# Scikit's mean-shift implementation using a flat kernel
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(feature_space)

# Cluster labels from the clustering
labels = mean_shift.labels_
labels = labels.reshape(height, width)


# Replace each pixel's value with the mean value of its belonging cluster
segmented_img = np.zeros_like(img)
for cluster in np.unique(labels):
    segmented_img[labels == cluster] = np.mean(img[labels == cluster], axis=0)


segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_LAB2BGR)
print(segmented_img.shape)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


