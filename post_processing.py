import numpy as np
import cv2

def post_processing(pred):
    '''
        pred is a numpy with HWC
    '''
    pad = np.zeros(((289), pred.shape[1]))

    pred = np.concatenate((pred, pad), axis=0)

    pred = pred[::-1, :]
    pred = pred.T

    pred = cv2.warpPolar(
        pred,
        (1250, 1250),
        (1250/2, 1250/2),
        1250//2,
        cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR
    )

    return pred