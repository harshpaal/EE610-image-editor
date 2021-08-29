import numpy as np  # used for handling array operations


class ImageProcessing(object):

    # defining convolution in image
    def convolution(self, image, window):

        window = np.flipud(np.fliplr(window))  # flipping the window
        output = self.correlation(image, window)  # computing the correlation
        output = np.uint8(output / window.sum())  # dividing the output with sum of window elements for normalizing

        return output  # return the computed image

    # computing histogram equalization of the image
    def histogram_equalization(self, image):

        image_row = image.shape[0]  # no. of rows of pixels in the image
        image_column = image.shape[1]  # no. of columns of pixels in the image

        # counting no. of values in the V channel of a HSV image matrix and setting minlength=256 to
        # ensure all 256 pixel values are covered and unavailable values are set to 0
        pmf = np.bincount(image.ravel(), minlength=256)

        pmf = pmf / (image_row * image_column)  # computing probability mass function(pmf)

        cdf = pmf.cumsum()  # computing cumulative distribution function(cdf) as cumulative sum of pmf

        # derive lookup for pixel values by multiplying cdf with 255 (max pixel value)
        # round the lookup to lower integer to avoid the pixel value 256
        pmf_lookup = np.uint8(np.floor(cdf * 255))

        output = image.copy()  # creating a copy of input image

        # for each pixel value replace the value with the corresponding value in lookup generated
        for r in range(256):
            output[image == r] = pmf_lookup[r]

        return output  # return the computed image

    # computing gamma correction of the image passed as ndarray based on gamma value passed
    def gamma_correction(self, image, gamma):

        normalization_const = 255.0 / np.float_power(255, gamma)  # calculating normalizing constant for image matrix
        output = np.uint8(normalization_const * np.float_power(image, gamma)) # s = C * r^gamma

        return output  # return the computed image

    # computing log transform of the image passed
    def log_transform(self, image):

        image_row = len(image)  # no. of rows in the image matrix
        image_column = len(image[0])  # no. of columns in the image matrix

        normalization_const = 255 / (np.log2(256))  # calculating normalizing constant for image matrix

        # s = C * log(r + 1)
        # 1 is added to input to avoid log(0)
        output = np.int8(normalization_const * np.log2(image + np.ones((image_row, image_column))))

        return output  # return the computed image

    # computing blurred image for the image and window size passed
    def blur(self, image, window_size):
        
        window = np.ones((window_size, window_size), dtype=np.uint8)  # matrix size = window size passed
        output = self.convolution(image, window)  # performing convolution of image and window
        
        return output  # return the computed image

    # computing sharpened image for the image and sharpening cost passed
    def sharp(self, image, sharp_const):

        window = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # defining standard 3x3 Laplacian window

        output = image - sharp_const * self.correlation(image, window)  # g(x,y) = f(x,y) - const*laplacian

        # keeping output pixels in range 0 to 255
        output[output < 0] = 0  # replace pixels with value < 0 in output with 0
        output[output > 255] = 255  # replace pixels with value > 255 in output with 255

        return output  # return the computed image

    # computing edges of the image passed using laplacian filter
    def edge_detection(self, image):

        window = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # defining window as standard 3x3 Laplacian
        output = self.correlation(image, window)  # performing correlation of image and window

        # keeping output pixels in range 0 to 255
        output[output < 0] = 0  # replace pixels with value < 0 in output with 0
        output[output > 255] = 255  # replace pixels with value > 255 in output with 255

        return output  # return the computed image

    # computing correlation operation on image and window passed
    def correlation(self, image, window):

        output = np.zeros_like(image)  # returns zero ndarray of same shape as of the image
        image_row = image.shape[0]  # no. of rows of pixels in the image
        image_column = image.shape[1]  # no. of columns of pixels in the image

        window_size = window.shape[0]  # computing window size from window passed
        zero_padding = window_size - 1  # computing zero padding requirements
        offset = int(zero_padding / 2)  # computing offset to be used during convolution

        # creating the zero padded image
        image_zero_padded = np.zeros((image_row + zero_padding, image_column + zero_padding))
        image_zero_padded[offset:(-1 * offset), offset:(-1 * offset)] = image

        # computing correlation as shifted sum of image elements keeping window stationary
        for r in range(window_size):
            for c in range(window_size):
                output = output + window[r][c] * image_zero_padded[r:r + image_row, c:c + image_column]

        return output  # return the computed image
