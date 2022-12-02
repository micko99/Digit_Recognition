import os
from localization_inline_numbers import localization
from classification import Classification

c = Classification('cnn')

def load_image_and_return_number(filename):
    '''
        load image from flask, execute AI problem and return result
    '''
    localization(filename)
    if c.type == 'cnn':
        if os.path.exists('digits_final.model'):
            return c.predict_model()
        else:
            return 'Model does not exist'
    else:
        if os.path.exists('digits_final_nn.model'):
            return c.predict_model()
        else:
            return 'Model does not exist'