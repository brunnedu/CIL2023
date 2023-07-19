from sklearn.metrics import f1_score
import torch

'''
    Metrics taking two boolean masks as the input
'''

def jaccard(y_pred_b, y_true_b):
    tp = (y_true_b & y_pred_b).sum()
    p = (y_true_b | y_pred_b).sum()
    return  tp / p

class ThresholdBinaryF1Score():
    def __init__(self, cutoff = 0.5):
        super().__init__()
        '''
            Binarizes a prediction using a cutoff threshold and then computes the F1 score
        '''
        self.cutoff = cutoff

    def __call__(self, y_pred, y_true_b):
        y_pred_b = (y_pred >= self.cutoff) * 1.0
        return torch.Tensor([f1_score(y_true_b.flatten(), y_pred_b.flatten())])

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"