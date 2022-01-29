from skimage import feature
from imageio import imread
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks, rotate
import numpy as np

image0 = imread("/Users/alexa/Documents/UBCO/COSC499/medical-imaging-matching/test_images_kaggle/images/2016_BC016983_ CC_L.jpg")

# canny edge detection (show muscle boundary) 
image = feature.canny(image0[:,:,1], sigma=0)

# https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)

# Generating figure 1
fig, axes = plt.subplots(1, 5, figsize=(10, 6))
ax = axes.ravel()

ax[0].imshow(image0, cmap=cm.gray)
ax[0].set_title('Orig')
ax[0].set_axis_off()

ax[1].imshow(image, cmap=cm.gray)
ax[1].set_title('Edge detection')
ax[1].set_axis_off()


angle_step = 0.5 * np.diff(theta).mean()
d_step = 0.5 * np.diff(d).mean()
bounds = [np.rad2deg(theta[0] - angle_step),
          np.rad2deg(theta[-1] + angle_step),
          d[-1] + d_step, d[0] - d_step]
ax[2].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
ax[2].set_title('Hough transform')
ax[2].set_xlabel('Angles (degrees)')
ax[2].set_ylabel('Distance (pixels)')
ax[2].axis('image')

ax[3].imshow(image, cmap=cm.gray)
ax[3].set_ylim((image.shape[0], 0))
ax[3].set_axis_off()
ax[3].set_title('Detected lines')

peaks = [p for p in zip(*hough_line_peaks(h, theta, d))]
_, angle, dist = peaks[0]
# The origin is the top left corner of the original image. X and Y axis are horizontal and vertical edges respectively.
# The distance is the minimal algebraic distance from the origin to the detected line.

#print(angle, dist)
(x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
slope = np.tan(angle + np.pi/2)
#print((x0, y0, angle, slope))

print("before:")
print(ax[3])
ax[3].axline((x0+1500, y0), slope=slope, linewidth=1, color='r')
print("after")
print(ax[3])


# rotate and crop
image = rotate(image0, angle*180/np.pi, center=(0,0), resize=True)
print(dist)
image = image[:,int(dist):]  
ax[4].imshow(image, cmap=cm.gray)

plt.tight_layout()
plt.show()