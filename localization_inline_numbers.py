import numpy as np
import cv2
import os

def histogram(img, axis):
    '''
        Histogram projection implementation for localization numbers in image
    '''
    _, threshold_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if axis == 1:
        kernel = np.ones((1, 50))*255
    else:
        kernel = np.ones((50, 1))*255
    
    dilate_image = cv2.dilate(threshold_image, kernel, iterations = 1)
    
    signal = np.sum(dilate_image, axis)/255
    square_signal = np.zeros_like(signal)
    treshold = np.mean(signal) 
    square_signal[signal >= treshold] = 255
        
    help_1 = square_signal[:len(square_signal) - 1]
    help_2 = square_signal[1:len(square_signal)]
    help_3 = help_1 - help_2

    coordinates = [i for i in range(len(help_1)-1) if help_3[i] != 0]
    return coordinates
    
def localization(filename):
    '''
        Save regions of interest(ROI)
    '''
    print(filename)
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coordinates_x = histogram(gray_image, 0) # vertical histogram projection
    coordinates_y = histogram(gray_image, 1) # horizontal histogram projection

    y_begin = coordinates_y[0]
    y_end = coordinates_y[-1]

    ROI_number = 0
    i = 0
    path= 'images'
    while i in range(len(coordinates_x) - 1):
        x_begin = coordinates_x[i]
        x_end = coordinates_x[i + 1]
        roi = image[y_begin:y_end, x_begin:x_end]
        roi = cv2.copyMakeBorder(roi, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
        cv2.imwrite(os.path.join(path, 'ROI_{}.png'.format(ROI_number)), roi)
        i += 2
        ROI_number += 1

