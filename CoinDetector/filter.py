import numpy
from scipy import signal


# converts input image to grayscale
def grayscale(img):

    return numpy.array(0.25 * img[:, :, 0] + 0.5 * img[:, :, 1] + 0.25 * img[:, :, 2])


# applies sobel filter to a grayscale image
def sobel(img):

    # sobel filters for X and Y axis
    gx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = numpy.array([[-1, - 2, - 1], [0, 0, 0], [1, 2, 1]])

    # convolution
    gradient_x = signal.convolve2d(img, gx, mode='same')
    gradient_y = signal.convolve2d(img, gy, mode='same')

    return numpy.abs(gradient_x) + numpy.abs(gradient_y)


# applies gaussian filter to a grayscale image
def gaussian_filter(img, kernel_radius, sigma):

    two_sigma_squared = 2*sigma ** 2

    gaussian = numpy.ones((2*kernel_radius + 1, 2*kernel_radius + 1))
    weight_sum = 0.0

    for x in range(-kernel_radius, kernel_radius):
        for y in range(-kernel_radius, kernel_radius):

            # gaussian function
            weight = numpy.exp(-(x ** 2 + y ** 2)/two_sigma_squared) / (two_sigma_squared * numpy.pi)
            gaussian[x + kernel_radius, y + kernel_radius] = weight

            # sum the coefficient to do a normalization later
            weight_sum += weight

    # normalize due to energy loss
    gaussian = gaussian / weight_sum

    # convolved image
    return signal.convolve2d(img, gaussian, mode='same')


# converts a grayscale image to black and white
def black_and_white(img):

    height, width = img.shape
    bnw = numpy.zeros((height, width))
    max_value = numpy.max(img)
    for y in range(0, height):
        for x in range(0, width):
            if img[y, x] / max_value > 0.25:
                bnw[y, x] = 1

    return bnw

