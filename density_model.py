import torch
import torch.nn as nn
import torch.optim as optim

# Define the velocity field (Neural Network)
class VelocityField(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, t, h):
        return self.net(h)  # Smooth transformation

class DensityModel(nn.Module):
    def __init__(self, pre_slices, reference_slice, post_slices):
        super(DensityModel, self).__init__()
        self.pre_slices = pre_slices
        self.reference_slice = reference_slice
        self.post_slices = post_slices

        if len(post_slices)!=1 or len(pre_slices)!=1:
            raise NotImplementedError
        
        self.velocity_field = VelocityField(hidden_dim=32)


    def fit(self):
        pass


    def get_location_distr(self, index):
        pass