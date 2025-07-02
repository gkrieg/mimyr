import torch
import torch.nn as nn
import anndata as ad
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from scipy.spatial import Delaunay
import numpy as np

def manual_alpha_shape_polygon(coords: np.ndarray, alpha: float) -> Polygon:
    tri = Delaunay(coords)

    def triangle_circumradius(a, b, c):
        A = np.linalg.norm(b - c)
        B = np.linalg.norm(c - a)
        C = np.linalg.norm(a - b)
        s = (A + B + C) / 2
        area = np.sqrt(max(s * (s - A) * (s - B) * (s - C), 1e-12))
        return (A * B * C) / (4.0 * area)

    edges = set()
    for simplex in tri.simplices:
        pts = coords[simplex]
        r = triangle_circumradius(pts[0], pts[1], pts[2])
        if r < 1.0 / alpha:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                edge = tuple(sorted((simplex[i], simplex[j])))
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)

    edge_lines = [LineString([coords[i], coords[j]]) for i, j in edges]
    boundary = unary_union(edge_lines)
    polygons = list(polygonize(boundary))

    if len(polygons) == 0:
        raise ValueError("Alpha shape produced no polygons.")
    elif len(polygons) == 1:
        return polygons[0]
    else:
        # Return the largest polygon (in area)
        return max(polygons, key=lambda p: p.area)



class KDEMixture(nn.Module):
    def __init__(self, spatial_data, bandwidth=1):
        super(KDEMixture, self).__init__()
        self.N, self.d = spatial_data.shape
        if not isinstance(spatial_data, torch.Tensor):
            spatial_data = torch.tensor(spatial_data).float()
        self.spatial_data = nn.Parameter(spatial_data, requires_grad=True)
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth, device=spatial_data.device), requires_grad=False)
        self.register_buffer("weights", torch.ones(self.N, device=spatial_data.device) / self.N)

    def forward(self, points, sample_frac=1.0):
        device = points.device
        if sample_frac < 1.0:
            k = max(1, int(self.N * sample_frac))
            idx = torch.randperm(self.N, device=self.spatial_data.device)[:k]
            spatial_subset = self.spatial_data[idx].to(device)
            weight_subset = self.weights[idx].to(device)
            weight_subset /= weight_subset.sum()
        else:
            spatial_subset = self.spatial_data.to(device)
            weight_subset = self.weights.to(device)

        distances = torch.cdist(points, spatial_subset)
        kernels = torch.exp(-distances ** 2 / (2 * self.bandwidth.to(device) ** 2))
        return torch.sum(weight_subset * kernels, dim=-1)


    def log_prob(self, points, sample_frac=1.0):
        density = self.forward(points, sample_frac=sample_frac)
        return torch.log(density + 1e-10)

    def prob(self, points, sample_frac=1.0):
        return self.forward(points, sample_frac=sample_frac)

    def sample(self, num_samples):
        indices = torch.randint(0, self.N, (num_samples,), device=self.spatial_data.device)
        noise = torch.randn(num_samples, self.d, device=self.spatial_data.device) * self.bandwidth
        return self.spatial_data[indices] + noise


class BiologicalModel2():
    def __init__(self, slices, z_posn=None, pin_key="region", bandwidth=0.001, z_factor=1):
        super(BiologicalModel2, self).__init__()
        self.bandwidth=bandwidth
        self.z_factor=z_factor
        try:
            self.xyz=ad.concat(slices).obsm["aligned_spatial"]
        except:
            self.slices = slices
            self.pin_key=pin_key
            self.z_posns = z_posn
            concatenated_slices = ad.concat(self.slices)
            z_posns_all=[]
            z_posns=self.z_posns
            for i,slice in enumerate(self.slices):
                slice.obsm["3d_spatial"]=np.concatenate([slice.obsm["aligned_spatial"],np.array([[z_posns[i]]]*len(slice))],1)
                z_posns_all.append(torch.tensor([z_posns[i]]*len(slice)).cuda())
            xy=torch.tensor(concatenated_slices.obsm["aligned_spatial"]).cuda()
            z_posns=torch.cat(z_posns_all)*self.z_factor
            self.xyz=torch.cat([xy,z_posns.unsqueeze(-1)],1)

    def fit(self):
        self.model=KDEMixture(self.xyz,bandwidth=self.bandwidth)
    
    
    def sample(self,z,size=1000):
        samples=[]
        while len(samples)<size:
            data_points=self.model.sample(100000)
            selection=torch.abs(data_points[:,2]-z)<self.z_factor/2
            samples+=data_points[selection].detach().cpu().numpy().tolist()
        return np.array(samples)

    def sample_slice_conditionally(self, slice, size=1000, alpha=1.0):
        coords_2d = slice.obsm["aligned_spatial"]

        z = slice.obsm["aligned_spatial"][:,2].mean().item()
        
        # Get shapely polygon
        polygon = manual_alpha_shape_polygon(coords_2d[:,:2], alpha=alpha)

        slice_xyz = torch.tensor(slice.obsm["aligned_spatial"], dtype=torch.float32).cuda()
        # Build KDE for the slice
        slice_kde = KDEMixture(slice_xyz, bandwidth=self.bandwidth)

        accepted = []
        while len(accepted) < size:
            candidates = self.model.sample(100000)

            z_mask = torch.abs(candidates[:, 2] - z) < self.z_factor
            candidates = candidates[z_mask]
            if len(candidates) == 0:
                continue

            candidates_2d = candidates[:, :2].detach().cpu().numpy()
            outside_mask = np.array([not polygon.contains(Point(pt)) for pt in candidates_2d])
            candidates = candidates[outside_mask]

            if len(candidates) == 0:
                continue

            accepted=accepted+candidates.detach().cpu().numpy().tolist()

        return np.array(accepted)[:size]



# import torchdiffeq
# import torch

# t_span = torch.tensor([0, 1], dtype=torch.float32).to("cuda")
# slice=bio_model.pre_slices[0]
# og_coords = slice.obsm["spatial"]

# # Convert numpy array to torch tensor and move to GPU
# transformed_coords = torch.tensor(og_coords, dtype=torch.float32).to("cuda")

# transformed_coords=density_model.alignment_velocity_fields[0].aff(transformed_coords)+torchdiffeq.odeint(density_model.alignment_velocity_fields[0], transformed_coords, t_span, method="rk4")[-1]                       
# nc=transformed_coords.cpu().detach().numpy()

# plt.scatter(nc[:,0],nc[:,1])