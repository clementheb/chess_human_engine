import numpy as np
from sklearn.metrics import accuracy_score



def extract_topk(array, k, axis, values: bool = False):
    """
    Values=True returns the values instead of the index of the top k values in array
    Does not work with arrays including values below -1
    """
    arr = array.copy()
    indices = []
    for _ in range(k):
        cur_idx = np.argmax(arr, axis=axis)
        indices.append(cur_idx)
        np.put_along_axis(arr, np.expand_dims(cur_idx, axis=axis), -1, axis=axis)
    idx_stacked = np.stack(indices, axis=axis)
    if values:
        return np.take_along_axis(array, idx_stacked, axis=axis)
    return idx_stacked


def array_threshold(array, threshold, lower:bool=False, replacement_val:int=0):###TOUT CHANGER POUR PRENDRE UN MASK ET PAS LES MOVES DIRECT
    """
    Lower = True returns only values lower than thresold"""
    if lower:
        super_threshold_indices = array > threshold
    else:
        super_threshold_indices = array < threshold
    array[super_threshold_indices] = replacement_val
    return(array)



def topk_accuracy(pred,true,k, axis, threshold:float=0):
    """
    Axis is the axis over which the multiclass are distributed, and not the samples axis
    i.e. the axis over which the mean of accuracies is done
    k is the number of coordinates to go over (so k=0 doesn't work - also division by zero)
    Threshold bounds the values taken into account : lower than threshold isnt taken into account
    """
    total_acc = 0
    pred = array_threshold(pred,threshold=threshold)
    true = array_threshold(true,threshold=threshold)

    for i in range(k):
        pred_temp = np.take(pred, i, axis=axis)
        true_temp = np.take(true, i, axis=axis)
        total_acc += accuracy_score(y_pred=pred_temp, y_true= true_temp)
    return(total_acc/k)


