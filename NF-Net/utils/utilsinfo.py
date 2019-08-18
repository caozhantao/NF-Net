import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

    

class statistic_type():
    TP = 1
    TN = 2
    FN = 3
    FP = 4

    statistic_type_max = 10


EXING_FLAG = 1
LIANG_FLAG = 0

def statistics_result(pre, label, statistics_dict={}):
    
    for i in range(0, len(pre)):
        if 1 == pre[i]:    
            if label.data[i] == EXING_FLAG:
                statistics_dict[statistic_type.TP] += 1
            else:
                statistics_dict[statistic_type.FP] += 1

        else:
            if label.data[i] == LIANG_FLAG:
                statistics_dict[statistic_type.TN] += 1
            else:
                statistics_dict[statistic_type.FN] += 1
            
    return statistics_dict
      
    



def compute_result(model_name, statistics_dict={}):
    TP = statistics_dict[statistic_type.TP]
    TN = statistics_dict[statistic_type.TN]
    FN = statistics_dict[statistic_type.FN]
    FP = statistics_dict[statistic_type.FP]

    Precision = 0.0
    Recall = 0.0
    F1Score = 0.0
    Acc = 0.0

    if (TP + FP) != 0:
        Precision = 1.0 * TP / (TP + FP)

    if (TP + FN) != 0:
        Recall = 1.0 * TP / (TP + FN)

    if (Recall + Precision) != 0:
        F1Score = 1.0 * 2 * Recall * Precision / (Recall + Precision)

    if (TP + TN + FP + FN) != 0:
        Acc = 1.0 * (TP + TN) / (TP + TN + FP + FN)
    
    print("%s, Precision:%f, Recall:%f, F1Score:%f, Acc:%f"% (model_name, Precision, Recall, F1Score, Acc))

