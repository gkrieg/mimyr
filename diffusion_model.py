import sys
from data_loader import SliceDataLoader


import itertools
import math
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import wandb   # <-- NEW

# ---------------- Polynomial features ----------------
def _all_exponent_tuples(input_dim: int, degree: int):
    exps = []
    for total in range(degree + 1):
        for bars in itertools.combinations_with_replacement(range(input_dim), total):
            e = [0] * input_dim
            for b in bars:
                e[b] += 1
            exps.append(tuple(e))
    return exps  # count = C(input_dim + degree, degree)


@dataclass
class TrainConfig:
    degree: int = 3
    hidden_sizes: tuple = (1024, 2048, 4096, 2048, 1024)
    activation: str = "relu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "poly"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 1000
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 4096
    lr: float = 2e-4
    weight_decay: float = 0
    epochs: int = 200
    grad_clip: float = None
    ema_decay: float = 0.999


# ---------------- Timestep embedding ----------------
class TimestepEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer(
            "freqs",
            torch.exp(torch.linspace(0, math.log(10000.0), steps=dim // 2))
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] scalar timesteps
        angles = t.float().unsqueeze(1) * self.freqs.view(1, -1)  # [B, dim//2]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


# ---------------- Featureizers ----------------
class PolynomialFeatures(nn.Module):
    def __init__(self, input_dim: int, degree: int):
        super().__init__()
        exps = _all_exponent_tuples(input_dim, degree)
        self.register_buffer("exponents", torch.tensor(exps, dtype=torch.long))

    @property
    def n_output_features_(self) -> int:
        return int(self.exponents.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)                  # [B,1,d]
        e = self.exponents.unsqueeze(0).to(x.device) # [1,T,d]
        monomials = torch.pow(x_expanded, e)
        return monomials.prod(dim=-1)


class IdentityFeatures(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @property
    def n_output_features_(self) -> int:
        return self.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RandomFourierFeatures(nn.Module):
    """
    RFF for RBF kernel k(x,y)=exp(-gamma * ||x-y||^2).

    W ~ N(0, 2*gamma * I),  b ~ Uniform[0, 2π).
    ϕ(x) = sqrt(2/D) * cos(x W^T + b)
    """
    def __init__(self, input_dim: int, num_features: int = 256, gamma: float = 1.0, seed: int | None = None):
        super().__init__()
        self._num_features = int(num_features)
        self.gamma = float(gamma)

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))

        W = torch.randn(self._num_features, input_dim, generator=g) * math.sqrt(2.0 * self.gamma)
        b = torch.rand(self._num_features, generator=g) * (2.0 * math.pi)

        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0 / self._num_features)

    @property
    def n_output_features_(self) -> int:
        return self._num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W.t() + self.b              # [B, D]
        return self.scale * torch.cos(proj)


# ---------------- Model ----------------
class NoisePredictor(nn.Module):
    def __init__(
        self,
        input_dim=3,
        degree=3,
        hidden_sizes=(512, 512),
        activation="silu",
        t_emb_dim=32,
        batchnorm=False,
        dropout=0.0,
        out_dim=3,
        feature_type="poly",
        num_rff_features=256,
        rff_gamma=1.0,
        rff_seed=None,
    ):
        super().__init__()

        if feature_type == "poly":
            self.featurizer = PolynomialFeatures(input_dim, degree)
        elif feature_type == "rff":
            self.featurizer = RandomFourierFeatures(
                input_dim=input_dim,
                num_features=num_rff_features,
                gamma=rff_gamma,
                seed=rff_seed,
            )
        elif feature_type == "none":            
            self.featurizer = IdentityFeatures(input_dim)
        else:
            raise ValueError("Only poly/none implemented here")

        feat_dim = self.featurizer.n_output_features_
        self.t_emb = TimestepEmbedding(dim=t_emb_dim)

        acts = {
            "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "leakyrelu": lambda: nn.LeakyReLU(0.1), "tanh": nn.Tanh,
        }
        Act = acts.get(activation.lower(), nn.SiLU)

        in_dim = feat_dim + t_emb_dim
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats_x = self.featurizer(x)
        t_emb = self.t_emb(t)
        feats = torch.cat([feats_x, t_emb], dim=-1)
        return self.net(feats)


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters()}
        for v in self.shadow.values():
            v.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.named_parameters():
            p.copy_(self.shadow[k])


# ---------------- Trainer ----------------
class DDPMTrainer:
    def __init__(self, coords_np: np.ndarray, cfg: TrainConfig):
        # assert coords_np.ndim == 2 and coords_np.shape[1] == 3
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        if coords_np is None:
            self.mean=np.array([[5.72180726, 3.86796997, 7.3023877 ]])
            self.std=np.array([[2.28143343, 1.64970914, 2.85336571]])
            print("Skipping data loading as coords_np is None")
        else:
            self.mean = coords_np.mean(axis=0, keepdims=True)
            self.std  = coords_np.std(axis=0, keepdims=True) + 1e-8

            x = (coords_np - self.mean) / self.std
            self.data = torch.tensor(x, dtype=torch.float32)
            dataset = TensorDataset(self.data)
            self.loader = DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
            )
            self.coords_np=coords_np

        # schedule builder
        def _build_schedules(cfg: TrainConfig):
            T = cfg.n_timesteps
            schedule_type = getattr(cfg, "schedule_type", "linear").lower()

            if schedule_type == "linear":
                betas = torch.linspace(
                    cfg.beta_start, cfg.beta_end, T, dtype=torch.float64
                ).clamp(1e-8, 0.999)
                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=0)

            elif schedule_type == "cosine":
                s = float(getattr(cfg, "cosine_s", 0.008))
                steps = torch.arange(T + 1, dtype=torch.float64)

                def f(u):
                    return torch.cos((u * math.pi / 2.0)).pow(2)

                u = ((steps / T)**3 + s) / (1.0 + s)
                abar = f(u)
                abar = abar / abar[0].clamp_min(1e-20)

                betas = (1.0 - (abar[1:] / abar[:-1]).clamp_min(1e-20)).clamp(1e-8, 0.999)
                alphas = 1.0 - betas
                alpha_bars = torch.cumprod(alphas, dim=0)

            else:
                raise ValueError("schedule_type must be 'linear' or 'cosine'")

            return (
                betas.to(torch.float32),
                alphas.to(torch.float32),
                alpha_bars.to(torch.float32),
            )

        betas, alphas, alpha_bars = _build_schedules(cfg)
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alpha_bars = alpha_bars.to(self.device)

        self.model = NoisePredictor(
            input_dim=3,
            degree=cfg.degree,
            hidden_sizes=cfg.hidden_sizes,
            activation=cfg.activation,
            batchnorm=cfg.batchnorm,
            dropout=cfg.dropout,
            out_dim=3,
            feature_type=cfg.feature_type,
        ).to(self.device)

        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.opt = optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )


    def train(self):
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            total_loss, n_batches = 0, 0

            for (x_batch,) in self.loader:
                # print("enter looop")
                x0 = x_batch.to(self.device)
                # print("x0",x0)
                B = x0.shape[0]
                t = torch.randint(0, self.cfg.n_timesteps, (B,), device=self.device)
                noise = torch.randn_like(x0)

                # print("noise",noise)

                alpha_bar_t = self.alpha_bars[t].view(-1, 1)
                x_noisy = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

                pred = self.model(x_noisy, t)
                loss = ((pred - noise) ** 2).mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                self.ema.update(self.model)

                total_loss += loss.item(); n_batches += 1

                # --- log per batch ---
                # wandb.log({"batch_loss": loss.item(), "epoch": epoch})

            avg_loss = total_loss / max(1, n_batches)
            print(f"Epoch {epoch:03d} | loss {avg_loss:.6f}")
            # wandb.log({"epoch_loss": avg_loss, "epoch": epoch}, commit=False)

            if epoch % 50 == 0:
                ckpt_name = f"/compute/oven-0-13/aj_checkpoints/full_ddpm_checkpoint_5_{epoch}.pt"
                if epoch % 200 == 0:
                    torch.save({
                        "model": self.model.state_dict(),
                        "ema": self.ema.shadow,
                        "opt": self.opt.state_dict(),
                        "cfg": self.cfg,
                    }, ckpt_name)
                # wandb.save(ckpt_name)

                # ---- generate and log a figure ----
                samples = self.sample(
                    len(slice_data_loader.train_slices[21]),
                    use_ema=False,
                    conditional_z=7.13971108,
                )
                plt.figure()
                plt.scatter(samples[:,0], samples[:,1], s=0.1, alpha=0.5)
                plt.xlim(0, 12)
                plt.ylim(0, 8)
                plt.title(f"Generated samples at epoch {epoch}")
                wandb.log({"generated_samples": wandb.Image(plt), "epoch": epoch})
                plt.close()

    @torch.no_grad()
    def sample(self, n_samples: int, use_ema=True, conditional_z=None) -> np.ndarray:

        if conditional_z is not None:
            conditional_z = (conditional_z - self.mean[0,2])/self.std[0,2]


        model = NoisePredictor(
            input_dim=3,
            degree=self.cfg.degree,
            hidden_sizes=self.cfg.hidden_sizes,
            activation=self.cfg.activation,
            batchnorm=self.cfg.batchnorm,
            dropout=self.cfg.dropout,
            out_dim=3,
            feature_type=self.cfg.feature_type,
        ).to(self.device)
        model.load_state_dict(self.model.state_dict())
        if use_ema: self.ema.copy_to(model)
        model.eval()

        # before the loop
        if conditional_z is not None:
            # shape [n_samples]
            if isinstance(conditional_z, (int, float)):
                cond_z_vec = torch.full((n_samples,), float(conditional_z), device=self.device)
            else:
                cond_z_vec = conditional_z.to(self.device).view(-1)
                assert cond_z_vec.shape[0] == n_samples, "conditional_z must be scalar or length n_samples"

            fixed_cond_noise = False  # set True if you want a fixed epsilon per sample across all t
            if fixed_cond_noise:
                cond_eps_fixed = torch.randn(n_samples, device=self.device)

        x = torch.randn(n_samples, 3, device=self.device)
        for t in reversed(range(self.cfg.n_timesteps-1)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
            if conditional_z is not None:
                if fixed_cond_noise:
                    eps = cond_eps_fixed
                else:
                    eps = torch.randn(n_samples, device=self.device)
                x[:, 2] = torch.sqrt(alpha_bar_t) * cond_z_vec + sqrt_one_minus_alpha_bar * eps            

            z = torch.randn_like(x) if t > 0 else 0
            pred_noise = model(x, torch.full((n_samples,), t, device=self.device))
            x = sqrt_recip_alpha * (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) + torch.sqrt(beta_t) * z

            # overwrite the conditioned dimension with the correctly noised value for this t
        if conditional_z is not None:
            x[:,2]= cond_z_vec

        x_np = x.cpu().numpy() * self.std + self.mean
        return x_np

    @torch.no_grad()
    def sample_with_guidance(
        self,
        n_samples: int,
        potential_model: nn.Module,
        guidance_scale: float = 1.0,
        use_ema: bool = True,
        conditional_z=None,
        conditional_x=None,
    ) -> np.ndarray:
        """
        Classifier/potential guidance sampling.
        potential_model: torch.nn.Module, takes coords [B,3] -> scalar potential [B]
        guidance_scale: float, multiplier on ∇ log p(y|x) ≈ ∇(-potential)
        """
        print("USing std and mean",self.std,self.mean)
        if conditional_z is not None:
            conditional_z = (conditional_z - self.mean[0,2]) / self.std[0,2]
        if conditional_x is not None:
            conditional_x = (conditional_x - self.mean[0,0]) / self.std[0,0]

        # restore model with EMA if needed
        model = NoisePredictor(
            input_dim=3,
            degree=self.cfg.degree,
            hidden_sizes=self.cfg.hidden_sizes,
            activation=self.cfg.activation,
            batchnorm=self.cfg.batchnorm,
            dropout=self.cfg.dropout,
            out_dim=3,
            feature_type=self.cfg.feature_type,
        ).to(self.device)
        model.load_state_dict(self.model.state_dict())
        if use_ema:
            self.ema.copy_to(model)
        model.eval()
        # potential_model = potential_model.to(self.device)
        # potential_model.eval()

        # initial Gaussian
        x = torch.randn(n_samples, 3, device=self.device)

        for t in reversed(range(self.cfg.n_timesteps-1)):
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            sqrt_recip_alpha = 1.0 / torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            # conditional_z overwrite
            if conditional_z is not None:
                eps = torch.randn(n_samples, device=self.device)
                x[:, 2] = torch.sqrt(alpha_bar_t) * conditional_z + sqrt_one_minus_alpha_bar * eps
            if conditional_x is not None:
                eps = torch.randn(n_samples, device=self.device)
                x[:, 0] = torch.sqrt(alpha_bar_t) * conditional_x + sqrt_one_minus_alpha_bar * eps


            # base prediction
            pred_noise = model(x, torch.full((n_samples,), t, device=self.device))

            # compute denoised sample estimate
            x0_pred = (x - sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_bar_t)

            # # --- guidance term ---
            # x0_pred.requires_grad_(True)
            # pot = potential_model(x0_pred, sample_frac=0.001).sum()  # scalar
            # grad = torch.autograd.grad(pot, x0_pred)[0]   # ∇ potential wrt clean sample
            # x0_pred = x0_pred.detach()

            grad = (potential_model(x0_pred* torch.tensor(self.std).to(x0_pred.device) + torch.tensor(self.mean).to(x0_pred.device))[1]) / torch.tensor(self.std).to(x0_pred.device)
            grad = grad.float()

            # inject gradient into noise estimate 
            guided_noise =  pred_noise - guidance_scale * grad

            # reverse update
            z = torch.randn_like(x) if t > 0 else 0
            x = sqrt_recip_alpha * (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * guided_noise) \
                + torch.sqrt(beta_t) * z

            # re-enforce condition on z-dimension
            if conditional_z is not None:
                x[:, 2] = conditional_z

            if conditional_x is not None:
                x[:, 0] = conditional_x

        x_np = x.detach().cpu().numpy() * self.std + self.mean
        return x_np
