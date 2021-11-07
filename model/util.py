import numpy as np
import torch

class Normalizer():
    def __init__(self, mini=None,maxi=None):
        self.mini = mini
        self.maxi = maxi
        
    def normalize_labels(self, labels, reset_min_max = False):
        ## added 0.001 for numerical stability
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))
        labels_norm = (labels - self.mini) / (self.maxi - self.mini)
        # Threshold labels <-- but why...
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)

        return labels_norm

    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini
#         return np.array(np.round(np.exp(labels) - 0.001), dtype=np.int64)
        return np.array(np.exp(labels) - 0.001)



def seed_everything():
    torch.manual_seed(0)
    import random
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False



def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)







