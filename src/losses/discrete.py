'''
    Losses taking two boolean masks as the input
'''

def IoU(y_pred_b, y_true_b):
    tp = (y_true_b & y_pred_b).sum()
    p = (y_true_b | y_pred_b).sum()
    return  tp / p
