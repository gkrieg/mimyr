import random
import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad
import torchdiffeq
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class KDEMixture(nn.Module):
    def __init__(self, spatial_data, bandwidth=10):
        super(KDEMixture, self).__init__()
        self.N, self.d = spatial_data.shape
        try:
            self.spatial_data = nn.Parameter(spatial_data, requires_grad=True)  
        except:
            spatial_data = torch.tensor(spatial_data).float()
            self.spatial_data = nn.Parameter(spatial_data, requires_grad=True)  
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth, device=spatial_data.device), requires_grad=False)
        self.register_buffer("weights", torch.ones(self.N, device=spatial_data.device) / self.N)

    def forward(self, points):
        """Compute KDE density estimates at given points."""
        distances = torch.cdist(points, self.spatial_data)  # Pairwise distances
        kernels = torch.exp(-distances ** 2 / (2 * self.bandwidth ** 2))
        return torch.sum(self.weights * kernels, dim=-1)

    def log_prob(self, points):
        """Compute log probability estimates at given points."""
        density = self.forward(points)
        return torch.log(density + 1e-10)

    def prob(self, points):
        """Compute log probability estimates at given points."""
        density = self.forward(points)
        return density

    def sample(self, num_samples):
        """Sample points from the KDE (approximate)."""
        indices = torch.randint(0, self.N, (num_samples,), device=self.spatial_data.device)
        noise = torch.randn(num_samples, self.d, device=self.spatial_data.device) * self.bandwidth
        return self.spatial_data[indices] + noise

class VelocityField(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        self.aff = nn.Linear(2,2, bias = True)
        # Initialize weights to identity
        self.aff.weight.data = torch.eye(2)

        # Initialize bias to zero
        self.aff.bias.data.zero_()

    def forward(self, t, h):
        return self.net(torch.cat((h, t.expand(h.shape[0], 1)), dim=1))

class AlignementModel():
    def __init__(self, slices, z_posn, pin_key="region", skip_alignemnt=False, use_ccf=False):
        super(AlignementModel, self).__init__()
        self.slices = slices
        self.z_posn = z_posn
        self.skip_alignemnt=skip_alignemnt
        self.pin_key=pin_key
        self.use_ccf=use_ccf

        concatenated_slices = ad.concat(slices)
        # self.pin_list = list(concatenated_slices.obs[self.pin_key].unique())


    def loss(self, reference_densitys, pinned_slice_densitys, velocity_field):
        t_span = torch.tensor([0, 1], dtype=torch.float32).to("cuda")
        loss=0

        j = random.randint(0,len(self.pin_list)-1)
        j2=random.randint(0,1)
        while len(pinned_slice_densitys[2*j+j2].spatial_data)<10 or len(self.current_slice.obsm["spatial"][self.current_slice.obs[self.pin_key]==self.pin_list[j]])<10:
            j = random.randint(0,len(self.pin_list)-1)
            j2=random.randint(0,1)

        points = self.current_slice.obsm["spatial"][self.current_slice.obs[self.pin_key]==self.pin_list[j]]


        spatial_coords = self.current_slice.obsm["spatial"]

        # Get min and max coordinates
        x_min, y_min = spatial_coords.min(axis=0)
        x_max, y_max = spatial_coords.max(axis=0)

        # Define resolution (adjust step size as needed)
        step_size = y_max/500  # Adjust grid resolution
        x_range = np.arange(x_min, x_max, step_size)
        y_range = np.arange(y_min, y_max, step_size)

        # Create mesh grid
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])


        if j2==0:
            points=points[points[:,0]<points.mean(0)[0]]
        else:
            points=points[points[:,0]>points.mean(0)[0]]


        original_ref_positions = torch.tensor(grid_points, dtype=torch.float32, requires_grad=True).cuda()
        new_ref_positions = velocity_field.aff(original_ref_positions)+torchdiffeq.odeint(velocity_field, original_ref_positions, t_span, method="rk4")[-1]
        # plt.scatter(x_ref_trans, -1*y_ref_trans, s=1, c="red", alpha=0.5, label="Slice without hole")        
        # plt.scatter(original_ref_positions.cpu().detach().numpy()[:,0], original_ref_positions.cpu().detach().numpy()[:,1], s=1, alpha=0.5, label="Slice with hole")
        # plt.scatter(new_ref_positions.cpu().detach().numpy()[:,0], new_ref_positions.cpu().detach().numpy()[:,1], s=1, alpha=0.5, label="Slice with hole")
        # plt.figure()

        p1 = reference_densitys[2*j+j2].prob(original_ref_positions)  
        p2 = pinned_slice_densitys[2*j+j2].prob(new_ref_positions)  

        loss=((p1-p2)**2).sum()        
               

        # loss = (new_ref_positions.mean() - pinned_slice_densitys[2*j+j2].spatial_data.mean())**2

        # Convert log-probabilities back to probabilities
        # prob1 = p1.exp()
        # prob2 = p2.exp()

        # # Normalize (zero mean)
        # prob1_centered = prob1 - prob1.mean()
        # prob2_centered = prob2 - prob2.mean()

        # # Compute Pearson correlation
        # corr = torch.sum(prob1_centered * prob2_centered) / (
        #     torch.sqrt(torch.sum(prob1_centered**2)) * torch.sqrt(torch.sum(prob2_centered**2))
        # )

        # # Loss: maximize correlation (minimize negative correlation)
        # loss = -corr
        
        # loss=((p1-p2)**6).sum()
        # kl1 = F.kl_div(p1.exp(), p2, reduction='batchmean', log_target=True)  # KL(p1 || p2)
        # kl2 = F.kl_div(p2.exp(), p1, reduction='batchmean', log_target=True)  # KL(p2 || p1)
        # loss = kl1 + kl2

        return loss


    def fit(self):
        if self.skip_alignemnt:
            for slice in self.slices:
                slice.obsm["aligned_spatial"] = slice.obsm["spatial"]
            return
        
        if self.use_ccf:
            for slice in self.slices:
                slice.obsm["aligned_spatial"] = np.stack([slice.obs["z_ccf"],slice.obs["y_ccf"],slice.obs["x_ccf"]],-1)
            return


        t_span = torch.tensor([0, 1], dtype=torch.float32).to("cuda")
        # reference_density = KDEMixture(self.reference_slice.obsm["spatial"])
        # reference_density.to("cuda")
        pinned_slice_densitys=[]

        all_slices=self.pre_slices+[self.reference_slice]+self.post_slices

        for i,slice in enumerate(all_slices):
            pinned_slice_densitys.append([])            
            for j,pin in enumerate(self.pin_list):
                pinned_slice=slice[slice.obs[self.pin_key]==pin]
                center=pinned_slice.obsm["spatial"].mean(0)[0]
                pinned_slice_1=pinned_slice[pinned_slice.obsm["spatial"][:,0]<center]
                pinned_slice_2=pinned_slice[pinned_slice.obsm["spatial"][:,0]>center]
                pinned_slice_densitys[-1].append(KDEMixture(pinned_slice_1.obsm["spatial"]))
                pinned_slice_densitys[-1][-1].to("cuda")
                pinned_slice_densitys[-1].append(KDEMixture(pinned_slice_2.obsm["spatial"]))
                pinned_slice_densitys[-1][-1].to("cuda")

        self.alignment_velocity_fields = []

        for i in range(len(pinned_slice_densitys)-1):
            reference_densitys=pinned_slice_densitys[i]
            next_densitys=pinned_slice_densitys[i+1]
            velocity_field = VelocityField(hidden_dim=128).cuda()
            self.current_slice=all_slices[i]
            
            optimizer = optim.Adam(velocity_field.parameters(), lr=0.01)

            for epoch in range(3001):
                if epoch % 1000 == 0:
                    with torch.no_grad():
                        x_to_fill = all_slices[i+1].obsm["spatial"][:,0]
                        y_to_fill = all_slices[i+1].obsm["spatial"][:,1]
                        
                        
                        transformed_batch_reference_coords = velocity_field.aff(torch.tensor(all_slices[i].obsm["spatial"]).float().cuda())+torchdiffeq.odeint(velocity_field, torch.tensor(all_slices[i].obsm["spatial"]).float().cuda(), t_span, method="rk4")[-1]                       
                        x_ref_trans=transformed_batch_reference_coords[:,0].cpu().numpy()
                        y_ref_trans=transformed_batch_reference_coords[:,1].cpu().numpy()


                        plt.figure(figsize=(8, 6))
                        plt.scatter(x_ref_trans, -1*y_ref_trans, s=1, c="red", alpha=0.5, label="Slice without hole")        
                        plt.scatter(x_to_fill, -1*y_to_fill, s=1, alpha=0.5, label="Slice with hole")
                        plt.title("Sample alignment mapping")
                        plt.legend()
                        # ct=0
                        # while os.path.exists(f"animations/{ct}_epoch_{epoch}.png"):
                        #     ct+=1
                        # plt.savefig(f"animations/{ct}_epoch_{epoch}.png")
                        plt.show()
                        # plt.show()
                            
                loss=self.loss(reference_densitys,next_densitys, velocity_field)
                loss.backward()
                if epoch%50==0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Epoch {epoch}, Loss: {loss}")

            self.alignment_velocity_fields.append(velocity_field)


    def get_common_coordinate_locations(self):
        if self.skip_alignemnt or self.use_ccf:
            return self.slices

        # Loop over all slices: pre_slices, reference_slice, and post_slices
        all_slices = self.pre_slices + [self.reference_slice] + self.post_slices

        # Iterate through each slice
        for i, slice in enumerate(all_slices):
            og_coords = slice.obsm["spatial"]

            # Convert numpy array to torch tensor and move to GPU
            transformed_coords = torch.tensor(og_coords, dtype=torch.float32).to("cuda")

            # If it's not the last slice, we need to apply the velocity fields
            if i < len(all_slices) - 1:
                # Iterate through the velocity fields for the current slice
                for j in range(i, len(all_slices) - 1):
                    velocity_field = self.alignment_velocity_fields[j]
                    t_span = torch.tensor([0, 1], dtype=torch.float32).to("cuda")
                    
                    # Perform ODE integration to map the coordinates using the velocity field
                    transformed_coords = velocity_field.aff(transformed_coords)+torchdiffeq.odeint(velocity_field, transformed_coords, t_span, method="rk4")[-1]

            # After transformation, add the aligned spatial coordinates to the slice
            slice.obsm["aligned_spatial"] = transformed_coords.cpu().detach().numpy()

        # Return the modified slices (pre_slices, reference_slice, post_slices)
        return self.pre_slices, self.reference_slice, self.post_slices



    def get_location_distr(self, index):
        pass





                # if epoch % 100 == 0:
            #     with torch.no_grad():
            #         x_to_fill = self.post_slices[0].obsm["spatial"][:,0]
            #         y_to_fill = self.post_slices[0].obsm["spatial"][:,1]

            #         transformed_batch_reference_coords = torchdiffeq.odeint(self.velocity_field, torch.tensor(self.reference_slice.obsm["spatial"]).float().cuda(), t_span, method="rk4")[-1]                       
            #         x_ref_trans=transformed_batch_reference_coords[:,0].cpu().numpy()
            #         y_ref_trans=transformed_batch_reference_coords[:,1].cpu().numpy()


            #         plt.figure(figsize=(8, 6))
            #         plt.scatter(x_ref_trans, -1*y_ref_trans, s=1, c="red", alpha=0.5, label="Slice without hole")        
            #         plt.scatter(x_to_fill, -1*y_to_fill, s=1, alpha=0.5, label="Slice with hole")
            #         plt.title("Sample alignment mapping")
            #         plt.legend()
            #         # ct=0
            #         # while os.path.exists(f"animations/{ct}_epoch_{epoch}.png"):
            #         #     ct+=1
            #         # plt.savefig(f"animations/{ct}_epoch_{epoch}.png")
            #         plt.show()
            #         # plt.show()
            