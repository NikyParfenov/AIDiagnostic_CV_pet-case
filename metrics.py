import numpy as np


def dice_coefficient(y_pred, y_true):
    
    intersection = 2.0 * (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()
    
    # We can proccess the exception or add some small value to denominator
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0

    return intersection / union
    
