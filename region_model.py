import torch
import torch.nn as nn
import torch.optim as optim

class RegionModel(nn.Module):
    def __init__(self, pre_slices, reference_slice, post_slices):
        super(RegionModel, self).__init__()
        self.pre_slices = pre_slices
        self.reference_slice = reference_slice
        self.post_slices = post_slices
            
    def fit(self):
        pass
    
    def get_region_distr(self, index):
        pass