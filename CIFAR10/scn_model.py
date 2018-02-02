# This script constructs the Sparse ConvNet model
# Written by Caesar

##################################################################################################################################################################
#
#                                                                   Import Section
#
##################################################################################################################################################################

import tensorflow as tf
import numpy as np
import utils as utl

##################################################################################################################################################################
#
#                                                                   Model Class Definition Section
#
##################################################################################################################################################################

class SCN_Model:

    """This class maps the Sparse ConvNet model and encapsulates several core functions."""

    def __init__(self, class_num, image_size, channel_num, k):

        """
        :param class_num: the total possible number of classes, for CIFAR problem this can be 10 or 100
        :param image_size: this parameter regulates input image as (image_size * image_size)
        :param channel_num: channel number of input image, 3 for RGB images
        :param k: the hyperparameter defined in the SCN paper, which is the filter number in the 1st Conv layer
        :return:
        """



