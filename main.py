# Author: Harsh Pal
# Github: harshpaal
# reference: in the report

# importing utility libraries
import sys  # used for parsing GUI arguments
import cv2  # used for reading/writing images and colorspace conversion

import matplotlib.pyplot as plt  # used for pop-up plots

# PyQt5 libraries are used for GUI
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QWidget, QMainWindow

# importing gui and image processing modules
from image_processing import *
from gui import *


# main GUI window class
class ImageEditorClass(QMainWindow):

    original_img = [0]  # storing original image for undoAll functionality

    current_img = [0]  # storing the current image for processing

    prev_img = [0]  # storing the previous image for use in Undo functionality

    img_blur = [0]  # storing copy of image being blurred

    img_sharp = [0]  # storing copy of image being sharpened

    # storing current image height and width
    img_width = 0
    img_height = 0

    img_object = ImageProcessing()  # initializing an object of ImageProcessing from imageProcessingFns.py

    current_code = -1  # storing code of current operation

    # codes of different operations
    # Histogram Equalization => 0
    # Gamma Correction => 1
    # Log Transform => 2
    # Negative => 3
    # Blur => 4
    # Sharpen => 5
    # Edge detection => 6

    # GUI initialization
    def __init__(self, parent=None):
        # initializing QWidget Qt module
        super(ImageEditorClass, self).__init__()
        QWidget.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # assigning functions to be called on all button clicked and slider events
        self.ui.openImageButton.clicked.connect(lambda: self.open_image())
        self.ui.saveImageButton.clicked.connect(lambda: self.save_image())

        self.ui.histogramEqualizationButton.clicked.connect(lambda: self.histogram_equalization())
        self.ui.logTransformButton.clicked.connect(lambda: self.log_transform())
        self.ui.gammaCorrectionButton.clicked.connect(lambda: self.gamma_correction())

        self.ui.blurExtendInputSlider.valueChanged.connect(lambda: self.blur())
        self.ui.sharpenExtendInputSlider.valueChanged.connect(lambda: self.sharpen())

        self.ui.undoButton.clicked.connect(lambda: self.undo())
        self.ui.undo_allButton.clicked.connect(lambda: self.undo_all())

        self.ui.viewHistogramButton.clicked.connect(lambda: self.view_histogram())
        self.ui.detectEdgeButton.clicked.connect(lambda: self.edge_detection())

        # initializes input dialog box gui for input of gamma value
        self.newDialog = InputDialogGuiClass(self)

    # called when Open button is clicked
    def open_image(self):
        self.set_default_slider()  # resetting blur and sharpen sliders to initial position

        open_image_window = QFileDialog()  # opens a new Open Image dialog box
        image_path = QFileDialog.getOpenFileName(open_image_window, 'Open Image', '/')  # capturing current image path

        # check if image path is not null or empty
        if image_path:
            # initialize class variables
            self.current_img = [0]
            self.current_code = -1

            # read image at selected path to a numpy ndarray object as color image
            path, _ = image_path
            self.current_img = cv2.imread(path, 1)

            # convert the image read to HSV format from default BGR format
            self.current_img = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2HSV)

            # set image specific class variables based on current image
            self.img_width = self.current_img.shape[1]
            self.img_height = self.current_img.shape[0]

            self.original_img = self.current_img.copy()
            self.prev_img = self.current_img.copy()

            self.display_image()  # converting current image from ndarry to pixmap and assigns it to image display label

            # enabling all buttons and sliders in the window.
            # Only Open button is enabled on start
            self.enable_options()

    # called when Save button is clicked
    def save_image(self):
        # configure the save image dialog box to use .jpg extension for image if not provided in file name
        dialog = QFileDialog()
        dialog.setDefaultSuffix('jpg')
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        # open the save dialog box and wait until user clicks 'Save' button in the dialog box
        if dialog.exec_() == QDialog.Accepted:

            save_image_filename = dialog.selectedFiles()[0]  # select the first path as image save location

            # write current image to the file path selected by user
            cv2.imwrite(save_image_filename,
                        cv2.cvtColor(self.current_img, cv2.COLOR_HSV2BGR))

    # called when Histogram Equalization button is clicked
    def histogram_equalization(self):
        self.update_previous_image()  # updating the previous image class variable with current image

        self.current_code = 0  # updating current operation code class variable

        self.set_default_slider()  # resetting blur and sharpen sliders to initial position

        # update V channel of the current image with histogram equallized matrix
        self.current_img[:, :, 2] = self.img_object.histogram_equalization(self.current_img[:, :, 2])

        self.display_image()  # converting current image from ndarry to pixmap and assigns it to image display label

    def gamma_correction(self):

        self.update_previous_image()  # updating the previous image class variable with current image

        self.current_code = 1  # update current operation code class variable

        self.set_default_slider()  # resetting blur and sharpen sliders to initial position

        # open gamma input dialog box and wait for dialog box to exit
        if self.newDialog.exec() == 0:
            gamma_value = self.newDialog.gamma  # read gamma value from gamma input dialog box class

            # reset the value of gamma in gamma input dialog box to 1
            self.newDialog.gammaInput.setText('1.00')
            self.newDialog.gamma = 1.0

            # perform gamma correction for positive gamma values gamma range is
            # restricted to 0 to 10 in the gamma input dialog box
            if gamma_value > 0:
                # update V channel of the current image with gamma corrected matrix
                self.current_img[:, :, 2] = self.img_object.gamma_correction(self.current_img[:, :, 2], gamma_value)

        self.display_image()  # converting current image from ndarry to pixmap and assigns it to image display label

    def log_transform(self):
        self.update_previous_image()  # updating the previous image class variable with current image

        self.current_code = 2  # update current operation code class variable

        self.set_default_slider()  # resetting blur and sharpen sliders to initial position

        # update V channel of the current image with log transformed matrix
        self.current_img[:, :, 2] = self.img_object.log_transform(self.current_img[:, :, 2])

        self.display_image()  # converting current image from ndarry to pixmap and assigns it to image display label

    def blur(self):
        self.update_previous_image()  # updating the previous image class variable with current image

        # disconnect, initialize and reconnect the sharpen slider valuechanged event
        # this is to avoid calling of sharpen function when sharpen slider value is reset
        self.ui.sharpenExtendInputSlider.valueChanged.disconnect()
        self.ui.sharpenExtendInputSlider.setValue(0)
        self.ui.sharpenExtendInputSlider.valueChanged.connect(lambda: self.sharpen())
        self.ui.sharpenValueLabel.setText('0')

        # read current blur value from slider and compute blur window size as (2 * slider value + 1)
        blur_value = int(np.floor(self.ui.blurExtendInputSlider.value()))
        blur_window_size = (blur_value * 2) + 1

        # if the operation being performed currently is blur, use initial image passed to blur function
        # else set current image as initial image for blur
        if self.current_code == 4:
            self.current_img = self.img_blur.copy()
        else:
            self.img_blur = self.current_img.copy()

        if blur_value > 0:
            self.ui.undoButton.setEnabled(True)  # enable undo button

            # update V channel of the current image with blurred V matrix
            self.current_img[:, :, 2] = self.img_object.blur(self.current_img[:, :, 2], blur_window_size)

        self.current_code = 4  # update current operation code class variable

        self.ui.blurValueLabel.setText(str(blur_value))
        self.display_image()

    def sharpen(self):
        self.update_previous_image()  # updating the previous image class variable with current image

        # disconnect, initialize and reconnect the blur slider value changed event
        # this is to avoid calling of blur function when blur slider value is reset
        self.ui.blurExtendInputSlider.valueChanged.disconnect()
        self.ui.blurExtendInputSlider.setValue(0)
        self.ui.blurExtendInputSlider.valueChanged. \
            connect(lambda: self.blur())
        self.ui.blurValueLabel.setText('0')

        sharpen_value = self.ui.sharpenExtendInputSlider.value()  # read current sharpen value from slider
        sharpen_const = sharpen_value / 10.0  # compute sharpen constant as (slider value/10)

        # if the operation being performed currently is sharpen, use initial image passed to sharpen function
        # else set current image as initial image for sharpen
        if self.current_code == 5:
            self.current_img = self.img_sharp.copy()
        else:
            self.img_sharp = self.current_img.copy()

        if sharpen_const > 0:
            self.ui.undoButton.setEnabled(True)  # enable undo button

            # update V channel of the current image with sharpened V channel matrix
            self.current_img[:, :, 2] = np.uint8(self.img_object.sharp(self.current_img[:, :, 2], sharpen_const))

        self.current_code = 5  # update current operation code class variable

        self.ui.sharpenValueLabel.setText(str(sharpen_value))
        self.display_image()

    def undo(self):
        self.ui.undoButton.setEnabled(False)
        self.current_img = self.prev_img.copy()
        self.display_image()

    def undo_all(self):
        # resetting blur and sharpen sliders to initial position
        self.set_default_slider()
        self.current_img = self.original_img.copy()

        # converting current image from ndarry format to pixmap and assigns it to image display label
        self.display_image()
        self.ui.undoButton.setEnabled(False)

    def view_histogram(self):
        # count the no of values corresponding to each value in the V channel of
        # image matrix give a minimum length of 256 to the counting to ensure all 256 pixel
        # values are covered or pixel values not available in image are set to zero
        histogram = np.bincount(self.current_img[:, :, 2].ravel(), minlength=256)

        # start a new figure to show histogram - assign title and axes label
        plt.figure(num='Image Histogram')

        # assign a discrete plot of histogram to figure
        plt.stem(histogram)
        plt.xlabel('Intensity levels')
        plt.ylabel('No. of pixels')

        # show the stem plot
        plt.show()

    def edge_detection(self):
        self.update_previous_image()  # updating the previous image class variable with current image
        self.current_code = 6  # updating current operation code class variable

        self.set_default_slider()  # resetting blur and sharpen sliders to initial position

        # update V channel of the current image with edge detected V channel matrix
        self.current_img[:, :, 2] = self.img_object.edge_detection(self.current_img[:, :, 2])

        self.display_image()

    # display_image converts current image from ndarry format to pixmap and assigns it to image display label
    def display_image(self):
        display_size = self.ui.imageDisplayLabel.size()  # setting display size to size of the image display label

        image = np.array(self.current_img.copy())  # copying current image to temporary variable for processing pixmap
        zero = np.array([0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)  # convert HSV image to RGB format for display in label

            # ndarray cannot be directly converted to QPixmap format required by image display label so ndarray is
            # first converted to QImage and then QImage to QPixmap convert image ndarray to QImage format
            qImage = QImage(image, self.img_width, self.img_height,
                            self.img_width * 3, QImage.Format_RGB888)

            # converting QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.ui.imageDisplayLabel.setPixmap(pixmap)  # set pixmap to image display label in GUI

    # enable_options enable all buttons and sliders in the window. Only Open button is enabled on start
    # Undo button remains disabled until an operation is performed
    def enable_options(self):
        self.ui.histogramEqualizationButton.setEnabled(True)
        self.ui.gammaCorrectionButton.setEnabled(True)
        self.ui.logTransformButton.setEnabled(True)

        self.ui.blurExtendInputSlider.setEnabled(True)
        self.ui.sharpenExtendInputSlider.setEnabled(True)

        self.ui.saveImageButton.setEnabled(True)
        self.ui.undo_allButton.setEnabled(True)
        self.ui.undoButton.setEnabled(False)

        self.ui.viewHistogramButton.setEnabled(True)
        self.ui.detectEdgeButton.setEnabled(True)

    # resetting blur and sharpen sliders to initial position
    def set_default_slider(self):
        # disconnect the value changed event of sliders from the functions assigned this prevents calling the blur
        # and sharpen function on resetting of sliders
        self.ui.sharpenExtendInputSlider.valueChanged.disconnect()
        self.ui.blurExtendInputSlider.valueChanged.disconnect()

        # update slider values to initial position i.e. 0, update slider value labels to 0
        self.ui.blurExtendInputSlider.setValue(0)
        self.ui.blurValueLabel.setText('0')
        self.ui.sharpenExtendInputSlider.setValue(0)
        self.ui.sharpenValueLabel.setText('0')

        # reconnect the value changed event of sliders to blur and sharpen functions
        self.ui.blurExtendInputSlider.valueChanged.connect(lambda: self.blur())
        self.ui.sharpenExtendInputSlider.valueChanged.connect(lambda: self.sharpen())

        # reset values of blur and sharpen image class variables
        self.img_blur = [0]
        self.img_sharp = [0]

        # enable Undo button only if an operation was performed previosly
        # i.e. current operation code is a valid code
        if (self.current_code >= 0) and not self.ui.undoButton.isEnabled():
            self.ui.undoButton.setEnabled(True)

    # updating the previous image class variable with current image
    def update_previous_image(self):
        self.prev_img = self.current_img.copy()


# initialize the ImageEditorClass and run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = ImageEditorClass()
    myapp.showMaximized()
    sys.exit(app.exec_())
