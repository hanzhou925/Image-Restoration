# Image Restoration
**1. Introduction:**

Image is a very common information carrier, but the image may be affected by noise due to various reasons in the process of image acquisition, transmission and storage. How to remove the influence of noise and restore the original image information is an important research problem in computer vision.
Common image restoration algorithms include median filtering based on spatial domain, wavelet denoising based on wavelet domain, nonlinear diffusion filtering based on partial differential equation, etc. In this experiment, we want to add noise to the image, and add noise to the image based on linear regression model denoising.

MNIST is a handwritten digit data set. The training set contains 60,000 handwritten digits, and the test set contains 10,000 handwritten digits, with a total of 10 categories.
 
 
**2. Procedure:**

2.1 Generate damaged images:
a) The damaged image was obtained by adding different noise masks to the original image. 
b) The noise mask contains only {0,1} values. For the noise mask of the original image, each row can be generated with a noise ratio of 0.8/0.4/0.6, that is, the pixel value of 80/40/60% of each row of each channel of the noise mask is 0, and the pixel value of the others is 1.

2.2 Image Restoration with regional binary linear regression model

2.3 Evaluation:
The evaluation error is the sum of the 2-norm of all restored images and the original image, and the smaller the error, the better.


**3. Reference:**

OpenCV：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
Numpy：https://www.numpy.org/
