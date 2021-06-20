import numpy as np
import logging
from sklearn import metrics

from pytorch_utils import forward
from utilities import get_filename
import config

import torch
import torch.nn.functional as F

def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy

def clip_nll(y_true, y_score):
    loss = -np.mean(y_true * y_score)
    return loss

class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader, ALSO_RETURN_RAW=False):

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)
        
        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(clipwise_output, axis=-1), labels=None)
        accuracy = calculate_accuracy(target, clipwise_output)

        loss = clip_nll(target, clipwise_output)
        
        statistics = {
          'accuracy': accuracy,
          'loss':loss,
          'cm':cm,
        }
        
        if ALSO_RETURN_RAW:
          return statistics, output_dict

        return statistics
      
