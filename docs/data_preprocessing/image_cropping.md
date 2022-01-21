# Data Preprocessing

## Image Cropping

During our client meeting, Rasika mentioned that there may be some issues in the future depending on how the library we used crops/resizes the images. We need to ensure that our choice of library does not impede our model.

Currently, we are using `OpenCV` or `cv2` when importing to resize the images.

### OpenCV Image Resizing

`OpenCV` allows us to resize/scale images. There are several interpolation methods that `OpenCV` allows:

1. `cv2.INTER_AREA`: This is used when we need to shrink an image. Uses pixel area relation for resampling. When used for zooming it uses the `INTER_NEAREST` method.
2. `cv2.INTER_CUBIC`: This is slow but more efficient. This uses bicubic interpolation for resizing the image. When resizing and interpolating new pixels, this method acts on the 4x4 neighbouring pixels of an image. It then takes the weighted average of the 16 pixels to create the new interpolated pixel.
3. `cv2.INTER_LINEAR`: This is primarily used when zooming is required. This is the default interpolation technique in `OpenCV`. This method is somewhat similar to the `INTER_CUBIC` intepolation but it uses 2x2 neighoring pixels to get the weighted average for the interpolated pixel.
4. `cv2.INTERNEAREST`: This method uses the nearest neighbour concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.

Another thing we need to keep in mind is that we must be aware on if we need to keep in mind the original aspect ration of the image.

We also need to remember that when shrinking an image we need to resample the pixels.

As of now, it does not seem like there will be an issue other than varying image sizes and perhaps with aspect ratios. Currently, I would recommend sticking with `cv2.INTER_AREA` interpolation. However, I will be researching more and clarifying with Rasika.

### Sources

- [Image Resizing with OpenCV | LearnOpenCV](https://learnopencv.com/image-resizing-with-opencv/)
- [Image Resizing using OpenCV | Python - GeeksforGeeks](https://www.geeksforgeeks.org/image-resizing-using-opencv-python/)
- [OpenCV Resize image using cv2.resize() - TutorialKart](https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/)
