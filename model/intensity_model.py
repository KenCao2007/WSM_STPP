import math
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed

def _single_mvncdf(mu, cov, rect):
    a, b, c, d = rect
    rv = multivariate_normal(mean=mu, cov=cov)
    cdf_bd = rv.cdf([b, d])
    cdf_ad = rv.cdf([a, d])
    cdf_bc = rv.cdf([b, c])
    cdf_ac = rv.cdf([a, c])
    return cdf_bd - cdf_ad - cdf_bc + cdf_ac

def batch_mvncdf_torch(means, sigmas, rect, n_jobs=1):
    """
    Compute batch of 2D Gaussian CDFs over a rectangle using scipy, with PyTorch tensor inputs.

    Parameters:
        means:  (N, 2) torch.Tensor
        sigmas: (N, 2, 2) torch.Tensor
        rect:   (a, b, c, d): rectangle bounds
        n_jobs: int, number of parallel workers

    Returns:
        probs: (N,) torch.Tensor on CPU
    """
    # Detach and move to CPU numpy
    means_np = means
    sigmas_np = sigmas

    # Build covariance matrices
    covs_np = sigmas_np

    # Parallel computation
    probs_np = Parallel(n_jobs=n_jobs)(
        delayed(_single_mvncdf)(mu, cov, rect)
        for mu, cov in zip(means_np, covs_np)
    )

    return torch.tensor(probs_np, dtype=torch.float)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class smash_intensity_decouple_kernel(nn.Module):
    def __init__(self, num_units=64, cond_dim=0, num_types=1):
        super(smash_intensity_decouple_kernel, self).__init__()

        self.channels = 1
        self.cond_dim = cond_dim          # = d_model (single block dimension)
        self.num_types = num_types


        hidden = num_units
        d = cond_dim

        # =======================
        # 1. Time head (λ_T) module
        # =======================

        # Current time interval t embedding
        self.t_time_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        if d > 0:
            # Historical time embedding
            self.t_cond_temporal = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden)
            )
            # Historical spatial embedding
            self.t_cond_spatial = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden)
            )
            # Historical joint embedding
            self.t_cond_joint = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden)
            )
            # mark embedding (if any)
            if num_types > 1:
                self.t_cond_mark = nn.Sequential(
                    nn.Linear(d, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden)
                )
            else:
                self.t_cond_mark = None
        else:
            self.t_cond_temporal = None
            self.t_cond_spatial = None
            self.t_cond_joint = None
            self.t_cond_mark = None

        # Time head final output λ_T small MLP
        self.t_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

        # =========================
        # 2. Spatial head q_S module
        # =========================

        # Location s embedding, assume dimension is 2 (x,y)
        self.s_x_proj = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        # Time t embedding (spatial head also sees current t)
        self.s_t_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        if d > 0:
            # Historical time embedding
            self.s_cond_temporal = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden)
            )
            # Historical spatial embedding
            self.s_cond_spatial = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden)
            )
            # Historical joint embedding
            self.s_cond_joint = nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden)
            )
            # mark embedding (if any)
            if num_types > 1:
                self.s_cond_mark = nn.Sequential(
                    nn.Linear(d, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden)
                )
            else:
                self.s_cond_mark = None
        else:
            self.s_cond_temporal = None
            self.s_cond_spatial = None
            self.s_cond_joint = None
            self.s_cond_mark = None

        # Spatial head final output q_S small MLP (ensure non-negative)
        self.s_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )


    # ---------- Tool: split cond_flat ----------
    def _split_cond_flat(self, cond_flat):
        """
        cond_flat: [N, D_total]，split into (temporal, spatial, joint, mark)
        According to your convention: D_total = 3*d or 4*d
        """
        d = self.cond_dim
        if d <= 0 or cond_flat is None:
            return None, None, None, None

        D = cond_flat.shape[-1]
        assert D == 3 * d or D == 4 * d, \
            f"cond_dim mismatch: got {D}, expect 3*{d} or 4*{d}"

        cond_temporal = cond_flat[:, :d]
        cond_spatial = cond_flat[:, d:2 * d]
        cond_joint = cond_flat[:, 2 * d:3 * d]
        cond_mark = cond_flat[:, 3 * d:] if D == 4 * d else None
        return cond_temporal, cond_spatial, cond_joint, cond_mark

    # ---------- Tool: broadcast cond to target batch shape ----------
    def _align_cond_to_shape(self, cond, target_shape):
        """
        Align the first few dimensions of cond to target_shape (e.g. target_shape = (B, M))
        Supported typical cases:
          - cond: (B, L, D) and target_shape = (B, L)      -> return as is
          - cond: (B, 1, D) and target_shape = (B, M)      -> expand on dim=1 to (B, M, D)
          - cond: (B, D)    and target_shape = (B, M)       -> unsqueeze(1) then expand to (B, M, D)
        """
        if cond is None:
            return None

        # Case 1: ( ..., D ) already matches target_shape completely
        if cond.dim() == len(target_shape) + 1 and tuple(cond.shape[:-1]) == tuple(target_shape):
            return cond

        # Case 2: (B, 1, D) -> (B, M, D)
        if cond.dim() == len(target_shape) + 1:
            if (len(target_shape) == 2 and
                cond.shape[0] == target_shape[0] and
                cond.shape[1] == 1):
                return cond.expand(target_shape[0], target_shape[1], cond.shape[-1])

        # Case 3: (B, D) -> (B, M, D)
        if cond.dim() == len(target_shape):
            if len(target_shape) == 2 and cond.shape[0] == target_shape[0]:
                return cond.unsqueeze(1).expand(target_shape[0], target_shape[1], cond.shape[-1])

        raise ValueError(
            f"Cannot broadcast cond of shape {tuple(cond.shape)} "
            f"to target leading shape {target_shape}"
        )

    # ===============================
    # Time marginal intensity λ_T(t | history)
    # ===============================
    def get_tilde_intensity_t(self, t, cond):
        """
        t:    (..., 1), e.g. (B, L, 1) or (B, num_mc, 1)
        cond: Shape can be broadcast to t[...,0],
              Typical: (B, L, D), (B, 1, D) or (B, D), where D = 3*d or 4*d
        Returns: (..., 1), non-negative λ_T(t | history)
        """
        t_shape = t.shape[:-1]  # e.g. (B, L) or (B, num_mc)

        d = self.cond_dim
        cond_temporal = cond_spatial = cond_joint = cond_mark = None
        cond_flat = None

        if d > 0 and cond is not None:
            cond_broadcast = self._align_cond_to_shape(cond, t_shape)
            cond_flat = cond_broadcast.reshape(-1, cond_broadcast.shape[-1])
            cond_temporal, cond_spatial, cond_joint, cond_mark = self._split_cond_flat(cond_flat)

        t_flat = t.reshape(-1, 1)  # [N,1]

        # Current t embedding
        h = self.t_time_proj(t_flat)   # [N, hidden]

        if d > 0 and cond_flat is not None:
            # Historical time
            h_t = self.t_cond_temporal(cond_temporal)
            h = h + h_t

            # Historical space (if spatial_independent=False, add it)
            h_s = self.t_cond_spatial(cond_spatial)
            h_j = self.t_cond_joint(cond_joint)
            h = h + h_s + h_j

            # mark (if any)
            if self.t_cond_mark is not None and cond_mark is not None:
                h_m = self.t_cond_mark(cond_mark)
                h = h + h_m

        # Small MLP + softplus to get non-negative λ_T
        h = self.t_head(h)                 # [N, 1]
        lambda_T = F.softplus(h) + 1e-8    # Avoid 0
        lambda_T = lambda_T.reshape(*t_shape, 1)
        return lambda_T

    # ==========================================
    # Spatial conditional intensity q_S(s | t, history) (unnormalised)
    # ==========================================
    def get_intensity_s(self, x_spatial, x_temporal, cond):
        """
        x_spatial : (..., 2), e.g. (B, L, 2) or (B, num_mc, 2)
        x_temporal: (..., 1), shape and x_spatial first few dimensions are the same
        cond      : Shape can be broadcast to x_spatial[...,0],
                    Typical: (B, L, D), (B, 1, D) or (B, D), where D = 3*d or 4*d

        Returns:
            (..., 1), non-negative q_S(s | t, history)
        """
        spatial_shape = x_spatial.shape[:-1]   # e.g. (B, L) or (B, num_mc)

        d = self.cond_dim
        cond_temporal = cond_spatial = cond_joint = cond_mark = None
        cond_flat = None

        if d > 0 and cond is not None:
            cond_broadcast = self._align_cond_to_shape(cond, spatial_shape)
            cond_flat = cond_broadcast.reshape(-1, cond_broadcast.shape[-1])
            cond_temporal, cond_spatial, cond_joint, cond_mark = self._split_cond_flat(cond_flat)

        s_flat = x_spatial.reshape(-1, 2)      # [N,2]
        t_flat = x_temporal.reshape(-1, 1)     # [N,1]

        # Current s,t embedding
        h_s = self.s_x_proj(s_flat)   # [N, hidden]
        h_t = self.s_t_proj(t_flat)   # [N, hidden]
        h = h_s + h_t

        if d > 0 and cond_flat is not None:
            # Historical time
            h_ht = self.s_cond_temporal(cond_temporal)
            h = h + h_ht

            # Historical space (if spatial_independent=False, use it)
            h_hs = self.s_cond_spatial(cond_spatial)
            h_hj = self.s_cond_joint(cond_joint)
            h = h + h_hs + h_hj

            # mark (if any)
            if self.s_cond_mark is not None and cond_mark is not None:
                h_hm = self.s_cond_mark(cond_mark)
                h = h + h_hm

        # Small MLP output non-negative q_S
        h = self.s_head(h)                 # [N, 1]
        q_S = F.softplus(h) + 1e-8         # Ensure >0
        q_S = q_S.reshape(*spatial_shape, 1)
        return q_S

class smash_intensity_GMM_kernel(nn.Module):
    def __init__(self, num_units=64,cond_dim=0, num_types=1, K_trig=3):
        super(smash_intensity_GMM_kernel, self).__init__()

        self.channels = 1
        self.cond_dim = cond_dim          # = d_model
        self.num_types = num_types
        if num_types > 1:
            raise ValueError("num_types > 1 is not supported for GMM kernel")

        # ================== Structural hyperparameters ==================
        hidden = num_units
        d = cond_dim
        # Number of triggering Gaussians: control multi-peak ability (you can try 2,3,4)
        self.K_trig = K_trig

        # ======================
        # 1. Time part: λ_T(t|H)
        # ======================

        # Current time interval t feature
        self.time_feat = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        if d > 0:
            # Historical time embedding: use cond_temporal (+ mark) for summary
            self.time_hist_feat = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
        else:
            self.time_hist_feat = None

            # Extract trigger amplitude A and decay rate beta from historical features
        self.time_amp_head = nn.Linear(hidden, 1)   # Output A_raw
        self.time_beta_head = nn.Linear(hidden, 1)  # Output beta_raw

        # Global base intensity (≈ μ Z0), not dependent on history
        self.base_log = nn.Parameter(torch.zeros(1))

        # ======================
        # 2. Spatial part: q_S(s|t,H)
        # ======================

        # Parameters of background Gaussian g0^θ(s): mean + log std (diagonal covariance)
        self.base_mean = nn.Parameter(torch.zeros(2))          # ~ center0
        self.base_log_std = nn.Parameter(torch.zeros(2))       # ~ log sigma0_x, log sigma0_y

        # Scale of triggering Gaussian g2^θ(s; m_k) (all triggering peaks share the same diagonal covariance)
        self.trig_log_std = nn.Parameter(torch.zeros(2))       # ~ log sigma2_x, log sigma2_y

        if d > 0:
            # Use spatial embedding to predict the centers m_k(H) ∈ R^2 of K triggering peaks
            # Input use cond_spatial (pure spatial stream)
            self.trig_mean_head = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.K_trig * 2)   # K 个 (mu_x, mu_y)
            )

            # Use integrated cond (temporal + spatial + joint (+mark)) to output mixture weights
            hist_input_dim = d * (3 if num_types == 1 else 4)
            self.s_mix_mlp = nn.Sequential(
                nn.Linear(hist_input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.K_trig)       # K logits
            )
        else:
            self.trig_mean_head = None
            self.s_mix_mlp = None

    # ---------- Tool: split cond_flat ----------

    def _split_cond_flat(self, cond_flat):
        """
        cond_flat: [N, D_total], split into (temporal, spatial, joint, mark)
        According to your Transformer convention: D_total = 3*d or 4*d
        """
        d = self.cond_dim
        if d <= 0 or cond_flat is None:
            return None, None, None, None

        D = cond_flat.shape[-1]
        assert D == 3 * d or D == 4 * d, f"cond_dim mismatch: got {D}, expect 3*{d} or 4*{d}"

        cond_temporal = cond_flat[:, :d]
        cond_spatial = cond_flat[:, d:2 * d]
        cond_joint = cond_flat[:, 2 * d:3 * d]
        cond_mark = cond_flat[:, 3 * d:] if D == 4 * d else None
        return cond_temporal, cond_spatial, cond_joint, cond_mark

    # ---------- Tool: 2D Gaussian pdf, global mean ----------

    def _gaussian_pdf_2d_global_mean(self, x, mean, log_std):
        """
        2D Gaussian pdf, mean: [2] global, log_std: [2] global
        x: [N, 2]
        Returns: [N, 1]
        """
        std = torch.exp(log_std)           # [2]
        var = std * std                    # [2]

        diff = x - mean                    # [N, 2]
        mahal = (diff * diff) / var       # [N, 2]
        mahal = mahal.sum(dim=-1)         # [N]

        log_norm = -math.log(2.0 * math.pi) - log_std.sum()   # scalar
        log_pdf = log_norm - 0.5 * mahal                      # [N]
        return log_pdf.unsqueeze(-1)                          # [N,1]

    # ---------- Tool: 2D Gaussian pdf, diag cov, shared log_std ----------

    def _gaussian_pdf_2d_diag(self, x, mean, log_std):
        """
        2D Gaussian pdf, mean: [N,2], log_std: [2] global
        x: [N,2]
        Returns: [N,1]
        """
        std = torch.exp(log_std)           # [2]
        var = std * std                    # [2]

        diff = x - mean                    # [N,2]
        mahal = (diff * diff) / var       # [N,2]
        mahal = mahal.sum(dim=-1)         # [N]

        log_norm = -math.log(2.0 * math.pi) - log_std.sum()   # scalar
        log_pdf = log_norm - 0.5 * mahal                      # [N]
        return log_pdf.unsqueeze(-1)

    # ---------- Tool: components of time head ----------

    def _time_components(self, t_flat, cond_flat):
        """
        Given flattened t and cond, calculate
          base       ~ λ_base  >=0
          A_total    ~ trigger total amplitude >=0
          beta       ~ decay rate   >0
          lambda_T   = base + A_total * exp(-beta * t_flat)

        This corresponds to a "compressed version" approximation of λ_T(t) = μZ0 + Σ α e^{-β(t-t_i)}Z2(s_i) in gt.
        """
        # t_flat: [N,1]
        h_t = self.time_feat(t_flat)   # [N, hidden]

        d = self.cond_dim
        if d > 0 and cond_flat is not None:
            cond_temporal, cond_spatial, cond_joint, cond_mark = self._split_cond_flat(cond_flat)

            # Use cond_temporal (+mark) as time history representation
            hist_input = cond_temporal
            if cond_mark is not None and cond_mark.shape[1] == d:
                hist_input = hist_input + cond_mark

            h_hist = self.time_hist_feat(hist_input)   # [N, hidden]
            h = h_t + h_hist
        else:
            h = h_t

        # A_total, beta, base are all positive
        A_total = F.softplus(self.time_amp_head(h))          # [N,1]
        beta = F.softplus(self.time_beta_head(h)) + 1e-4     # [N,1]
        base = F.softplus(self.base_log)                     # scalar

        # λ_T(t) = base + A_total * exp(-beta * t)
        lambda_T_flat = base + A_total * torch.exp(-beta * torch.clamp(t_flat, min=0.0))

        return base, A_total, beta, lambda_T_flat


    def get_tilde_intensity_t(self, t, cond):
            """
            t:    (..., 1), e.g. (B, L, 1) or (B, num_mc, 1)
            cond: Shape can be broadcast to t[...,0],
                Typical: (B, L, D), (B, 1, D) or (B, D)
            Returns: (..., 1), non-negative λ_T(t|H_t)
            """
            # Target batch shape (without last dim 1)
            t_shape = t.shape[:-1]              # e.g. (B, L) or (B, num_mc)

            # Align cond, make cond.shape[:-1] == t_shape
            cond_broadcast = self._align_cond_to_shape(cond, t_shape)

            # flatten into [N, ...] then pass to _time_components
            t_flat = t.reshape(-1, 1)                              # [N,1]
            cond_flat = cond_broadcast.reshape(-1, cond_broadcast.shape[-1])  # [N,D]

            base, A_total, beta, lambda_T_flat = self._time_components(t_flat, cond_flat)

            lambda_T = lambda_T_flat.reshape(*t_shape, 1)
            return lambda_T

    def get_intensity_s(self, x_spatial, x_temporal, cond):
        """
        x_spatial : (..., 2), e.g. (B, L, 2) or (B, num_mc, 2)
        x_temporal: (..., 1), shape and x_spatial first few dimensions are the same
        cond      : Shape can be broadcast to x_spatial[...,0],
                    Typical: (B, L, D), (B, 1, D) or (B, D)

        Returns:
            (..., 1), unnormalised q_S(s | t, history)
        """
        # Target batch shape (without last dim 1)
        spatial_shape = x_spatial.shape[:-1]   # e.g. (B, L) or (B, num_mc)

        # Align cond, make cond.shape[:-1] == spatial_shape
        cond_broadcast = self._align_cond_to_shape(cond, spatial_shape)

        # flatten
        s_flat = x_spatial.reshape(-1, 2)                      # [N,2]
        t_flat = x_temporal.reshape(-1, 1)                     # [N,1]
        cond_flat = cond_broadcast.reshape(-1, cond_broadcast.shape[-1])  # [N,D]

        d = self.cond_dim
        if d > 0 and cond_flat is not None:
            cond_temporal, cond_spatial, cond_joint, cond_mark = self._split_cond_flat(cond_flat)
        else:
            cond_temporal = cond_spatial = cond_joint = cond_mark = None

        # First use time head to calculate base, A_total (keep consistent with λ_T)
        base, A_total, beta, lambda_T_flat = self._time_components(t_flat, cond_flat)
        # base: scalar；A_total: [N,1]

        # Background Gaussian g0(s)
        log_g0 = self._gaussian_pdf_2d_global_mean(
            s_flat, self.base_mean, self.base_log_std
        )
        g0 = torch.exp(log_g0)   # [N,1]



        # ---------- Multi-peak triggering part: K Gaussians ----------

        N = s_flat.shape[0]

        # 1) mixture weights π_k(H,t)
        mix_logits = self.s_mix_mlp(cond_flat)          # [N, K]
        mix_weights = F.softmax(mix_logits, dim=-1)     # [N, K]

        # 2) K triggering centers m_k(H)
        trig_means_flat = self.trig_mean_head(cond_spatial)      # [N, 2*K]
        trig_means = trig_means_flat.view(N, self.K_trig, 2)     # [N, K, 2]

        # 3) g2_k(s; m_k) density
        s_expanded = s_flat.unsqueeze(1)                         # [N,1,2]
        diff = s_expanded - trig_means                           # [N,K,2]

        std = torch.exp(self.trig_log_std)                       # [2]
        var = std * std                                          # [2]

        mahal = (diff * diff) / var                              # [N,K,2]
        mahal = mahal.sum(dim=-1)                                # [N,K]

        log_norm = -math.log(2.0 * math.pi) - self.trig_log_std.sum()
        log_g2_all = log_norm - 0.5 * mahal                      # [N,K]
        g2_all = torch.exp(log_g2_all)                           # [N,K]

        # 4) A_k = A_total * π_k
        A_total_broadcast = A_total.expand_as(mix_weights)       # [N,K]
        A_k = A_total_broadcast * mix_weights                    # [N,K]

        # 5) Triggering part: sum_k A_k * g2_k
        triggered = (A_k * g2_all).sum(dim=-1, keepdim=True)     # [N,1]

        # 6) Total spatial intensity
        q_flat = base * g0 + triggered + 1e-8                    # [N,1]

        q_S = q_flat.reshape(*spatial_shape, 1)
        return q_S



    def _align_cond_to_shape(self, cond, target_shape):
        """
        Align the first few dimensions of cond to target_shape (e.g. target_shape = (B, M))
        Supported typical cases:
          - cond: (B, L, D) and target_shape = (B, L)      -> return as is
          - cond: (B, 1, D) and target_shape = (B, M)      -> expand on dim=1 to (B, M, D)
          - cond: (B, D)    and target_shape = (B, M)       -> unsqueeze(1) then expand to (B, M, D)
        """
        if cond is None:
            return None

        # Case 1: ( ..., D ) already matches target_shape completely
        if cond.dim() == len(target_shape) + 1 and cond.shape[:-1] == target_shape:
            return cond

        # Case 2: (B, 1, D) -> (B, M, D)
        if cond.dim() == len(target_shape) + 1:
            if (len(target_shape) == 2 and
                cond.shape[0] == target_shape[0] and
                cond.shape[1] == 1):
                return cond.expand(target_shape[0], target_shape[1], cond.shape[-1])

        # Case 3: (B, D) -> (B, M, D)
        if cond.dim() == len(target_shape):
            if len(target_shape) == 2 and cond.shape[0] == target_shape[0]:
                return cond.unsqueeze(1).expand(target_shape[0], target_shape[1], cond.shape[-1])

        raise ValueError(
            f"Cannot broadcast cond of shape {tuple(cond.shape)} "
            f"to target leading shape {target_shape}"
        )



class smash_intensity_vanilla_kernel(nn.Module):
    def __init__(self, num_units=64,cond_dim=0, num_types=1):
        super(smash_intensity_vanilla_kernel, self).__init__()
        self.channels = 1
        self.cond_dim=cond_dim


        sinu_pos_emb = SinusoidalPosEmb(num_units)
        fourier_dim = num_units
        self.num_types = num_types

        time_dim = num_units
        

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.linears_spatial = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, num_units),
            ]
        )

        self.linears_temporal = nn.ModuleList(
            [
                nn.Linear(1, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, num_units),
            ]
        )

        self.intensity_t = nn.Sequential(
                nn.Linear(num_units, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, 1),
                nn.Softplus(beta=1)

        )
        self.intensity_s = nn.Sequential(
                nn.Linear(num_units * 2, num_units),
                nn.Softplus(beta = 1),
                nn.Linear(num_units, 1),
                nn.Softplus(beta=1)
        )

        self.linear_t = nn.Sequential(
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
        )

        self.linear_s = nn.Sequential(
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
        )



        self.cond_all = nn.Sequential(
                nn.Linear(cond_dim * 3 if num_types==1 else cond_dim * 4, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units)
        )

        self.cond_temporal = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units)
            ]
        )

        self.cond_spatial = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units)
            ]
        )

        self.cond_joint = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units),
                nn.Linear(cond_dim, num_units)
            ]
        )

        

    def get_tilde_intensity_t(self, t, cond):
        # this function returns the user-designed non-negative network that models 'unormalised intensity' of time
        # t is the time interval
        x_temporal = t


        hidden_dim = self.cond_dim

        cond_temporal, cond_spatial, cond_joint, cond_mark = cond[:,:,:hidden_dim], cond[:,:,hidden_dim:2*hidden_dim], cond[:,:,(2*hidden_dim):(3*hidden_dim)],cond[:,:,3*hidden_dim:] 

        
        cond = self.cond_all(cond)


        for idx in range(3):
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)

            cond_temporal_emb = self.cond_temporal[idx]((cond_temporal+cond_mark) if self.num_types>1 else cond_temporal)
            
            x_temporal += cond_joint_emb + cond_temporal_emb
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_temporal = self.linears_temporal[-1](x_temporal)

        pred = self.intensity_t(x_temporal)
        return pred

    def get_intensity_s(self, x_spatial, x_temporal,  cond):
        # this function computes energy function of location
        # x_temporal is the time interval
        hidden_dim = self.cond_dim

        cond_temporal, cond_spatial, cond_joint, cond_mark = cond[:,:,:hidden_dim], cond[:,:,hidden_dim:2*hidden_dim], cond[:,:,2*hidden_dim:3*hidden_dim],cond[:,:,3*hidden_dim:] 

        cond = self.cond_all(cond)

        alpha_s = F.softmax(self.linear_s(cond), dim=-1).squeeze(dim=1).unsqueeze(dim=2)
        alpha_t = F.softmax(self.linear_t(cond), dim=-1).squeeze(dim=1).unsqueeze(dim=2)


        for idx in range(3):
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)

            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_emb = self.cond_temporal[idx]((cond_temporal+cond_mark) if self.num_types>1 else cond_temporal)
            cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

            x_spatial += cond_joint_emb + cond_spatial_emb
            x_temporal += cond_joint_emb + cond_temporal_emb

            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_spatial = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)

        x_output_t = x_temporal * alpha_t[:,:1,:] + x_spatial * alpha_t[:,1:2,:]
        x_output_s = x_temporal * alpha_s[:,:1,:] + x_spatial * alpha_s[:,1:2,:]

        pred = self.intensity_s(torch.cat((x_output_t, x_output_s), dim=-1))
        return pred            



class smash_intensity_score(nn.Module):
    def __init__(self, num_units=64,cond_dim=0, num_types=1, T=10., S=[[-1,1],[-1,1]], K_trig=3, kernel_type='vanilla'):
        super(smash_intensity_score, self).__init__()
        self.channels = 1
        self.cond_dim = cond_dim
        self.T = T
        self.S = S
        self.num_types = num_types
        self.label_ce = nn.CrossEntropyLoss(reduction='none')

        if kernel_type == 'vanilla':
            self.kernel = smash_intensity_vanilla_kernel(num_units, cond_dim, num_types)
        elif kernel_type == 'decouple':
            self.kernel = smash_intensity_decouple_kernel(num_units, cond_dim, num_types)
        elif kernel_type == 'GMM':
            self.kernel = smash_intensity_GMM_kernel(num_units, cond_dim, num_types, K_trig)
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

        self.classifier = nn.Sequential(
                nn.Linear(3 * cond_dim if num_types==1 else 4 * cond_dim , num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
        )

        if num_types > 1:
            self.label_classifier = nn.Sequential(
                nn.Linear(4 * cond_dim + 3, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_types)
            )

    def get_tilde_intensity_t(self, t, cond):
        return self.kernel.get_tilde_intensity_t(t, cond)

    def get_intensity_s(self, s, t, cond):
        return self.kernel.get_intensity_s(s, t, cond)

    def get_ending_logit(self, cond):
        # this returns the probability of given cond, the probability that it ends here (then not ends here)
        logit = self.classifier(cond).squeeze()
        # prob = torch.sigmoid(logit)
        return logit

    def get_label_logits(self, t, loc, cond):
        # compute multi-class cross entropy loss for label prediction
        # input is current time, location and cond
        input = torch.cat((t, loc, cond), dim=-1)
        logit = self.label_classifier(input)
        return logit

    def get_c_s(self, t, cond, num_grid=5):
        # this function computes the normalising constant of location density
        # t is the time interval

        # compute integral in S
        grid_x = torch.linspace(self.S[0][0], self.S[0][1], num_grid).to(t.device)
        grid_y = torch.linspace(self.S[1][0], self.S[1][1], num_grid).to(t.device)
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y)

        bs = t.shape[0]
        grid = torch.stack((grid_xx.reshape(-1), grid_yy.reshape(-1)), dim=-1).unsqueeze(0).repeat(t.shape[0],1,1).reshape((-1, 1, 2))

        # expand t for num_grad*2 times
        t = t.repeat(1, num_grid**2, 1).reshape((-1,1,1))
        cond = cond.repeat(1, num_grid**2, 1).reshape((-1, 1, cond.shape[-1]))

        intensity = self.get_intensity_s(grid, t, cond).reshape((bs, num_grid**2, -1)).sum(dim=1)  # batch * num_types
        integral = intensity / (num_grid ** 2) * (self.S[0][1] - self.S[0][0]) * (self.S[1][1] - self.S[1][0]) # area of S
        return integral


    
    def get_tilde_Lambda_T(self, t, cond, num_grid=5):
        # This function computes \tilde \Lambda_n(T)
        # t is the actual event time, not time_gap
        T = self.T
        # first, for each element tau in t, we uniformly take num_grid value between 0 to [T-tau]

        T = torch.as_tensor(T, device=t.device, dtype=t.dtype).view(1,1,1)
        delta = (T - t).clamp_min(0).unsqueeze(1)                              # (bs,1,1,1)
        w = torch.linspace(0, 1, steps=num_grid, device=t.device, dtype=t.dtype).view(1, num_grid, 1, 1)  # (1,num_grid,1,1)
        out = (delta * w).reshape(-1, 1, 1)      
        cond_aug = cond.repeat((1, num_grid, 1)).reshape((-1, cond.shape[1], cond.shape[2]))
        intensity_grids = self.get_tilde_intensity_t(out, cond_aug)  
        interval_len = delta.squeeze(-1).repeat((1, num_grid, 1)).reshape((-1, 1, 1))
        cummulative_hazard = (intensity_grids * interval_len).reshape((-1, num_grid, 1)).sum(dim=1).unsqueeze(-1)

        return cummulative_hazard
    

    
    def get_tilde_Lambda(self, time_gap, cond, num_grid=5):
        # This function computes \tilde \Lambda_n
        # time_gap is the time gap between 

        dtype = time_gap.dtype
        steps = torch.linspace(0, 1, steps=num_grid+1, device=time_gap.device, dtype=dtype)  # (G,)

        out = (time_gap.to(dtype)).unsqueeze(1) * steps.view(1, -1, 1, 1)  # (bs, num_grid+1, 1, 1)
        t_aug = out.reshape(-1, 1, 1)  # (bs*(num_grid+1), 1, 1)
        cond_aug = cond.repeat((1, num_grid + 1, 1)).reshape((-1, cond.shape[1], cond.shape[2]))
        intensity_grids = self.get_tilde_intensity_t(t_aug, cond_aug)  # (总点数, 1)
        integral_grids = intensity_grids * time_gap.repeat((1, num_grid + 1, 1)).reshape((-1, 1, 1))
        Lambda = integral_grids.reshape(-1, num_grid + 1, 1, 1).sum(1) / num_grid  # (总点数, 1)
        return Lambda

    def get_score_s(self, s, t, cond, second_order=True):
        # t = self.get_gap(t)
        s_grad = torch.autograd.Variable(s, requires_grad=True)

        logq_s = (self.get_intensity_s(s_grad, t, cond) + 1e-10).log()
        score = torch.autograd.grad(logq_s.sum(), s_grad, create_graph=True, retain_graph=True)[0]
        # score = torch.autograd.grad(logq_s.sum(), s_grad, create_graph=True, retain_graph=True)[0]

        
        # the following code only for debugging
        # score = self.get_logintensity_s(s_grad, t, cond)
        if second_order:
            score_x_grad = torch.autograd.grad(score[:,:,0].sum(), s_grad, create_graph=True, retain_graph=True)[0][:,:, 0]
            score_y_grad = torch.autograd.grad(score[:,:,1].sum(), s_grad, create_graph=True, retain_graph=True)[0][:,:, 1]
            score_grad = torch.cat((score_x_grad, score_y_grad), dim=-1).unsqueeze(1)
        else:
            score_grad = None
        
        return score, score_grad

    def get_score_t(self, t, cond = None, second_order=True):
        # t = self.get_gap(t)
        t = torch.autograd.Variable(t, requires_grad=True)

        intensity = self.get_tilde_intensity_t(t, cond)
        intensity_log = (intensity+1e-10).log()

        intensity_grad_t = torch.autograd.grad(intensity_log.sum(), t, create_graph=True, retain_graph=True)[0]
        # intensity_grad_t = torch.autograd.grad(intensity_log.sum(), t, create_graph=True, retain_graph=True)[0]
        score_t = intensity_grad_t - intensity
        if second_order:
            score_t_grad = torch.autograd.grad(score_t.sum(), t, create_graph=True)[0]
        else:
            score_t_grad = None
        
        return score_t, score_t_grad
    

    def get_label_ll(self, t, event_loc, cond, mark):
        label_logits = self.get_label_logits(t, event_loc, cond)
        # mark_nonmask = self.depadding(mark, non_pad_mask)
        label_loss = -self.label_ce(label_logits.squeeze(), mark.squeeze().long())
        return label_loss.sum()
    
    def get_lls(self, event_loc, event_time, t, cond_expand, cond, end_nonmask, num_grid=5, num_grid_s=5, mark = None, with_survival=True, t_expand=None):
        # t is the time gap
        tilde_intensity_t = self.get_tilde_intensity_t(t, cond)
        
        qn = self.get_intensity_s(event_loc, t, cond)
        C_s_n = self.get_c_s(t, cond, num_grid_s).unsqueeze(-1)

        if not with_survival:
            Lambda_n = self.get_tilde_Lambda(t_expand, cond_expand, num_grid)
            t_ll = ((tilde_intensity_t + 1e-10).log().sum() - Lambda_n.sum())
        else:
            Lambda_n = self.get_tilde_Lambda(t, cond, num_grid)
            hat_Gn = torch.sigmoid(self.get_ending_logit(cond_expand)[:,0]).unsqueeze(-1).unsqueeze(-1)
            hat_Gn = torch.where(end_nonmask, 1-hat_Gn, hat_Gn)
            Lambda_n_T = self.get_tilde_Lambda_T(event_time, cond, num_grid)
            tilde_G_n_T = torch.exp(-Lambda_n_T)
            t_ll = ((tilde_intensity_t + 1e-10).log() - Lambda_n - (1 - tilde_G_n_T + 1e-10).log() ).sum() + (hat_Gn + 1e-10).log().sum()

        s_ll = (qn / (C_s_n + 1e-10) + 1e-10).log().sum()
        
        if self.num_types > 1:
            label_ll = self.get_label_ll(t, event_loc, cond, mark).sum()
            return t_ll, s_ll, label_ll
        else:
            return t_ll, s_ll




    def get_score(self, s, t, cond, second_order=True):

        score_s, score_grad_s = self.get_score_s(s, t, cond, second_order)
        score_t, score_grad_t = self.get_score_t(t, cond, second_order)

        return score_s, score_grad_s, score_t, score_grad_t


    def get_intensity(self, s, t, cond, with_survival=True, num_grid_t=10, num_grid_s=16):
        
        # t is the time_gap
        time_gap = t
        tilde_lambda = self.get_tilde_intensity_t(time_gap, cond)
        q = self.get_intensity_s(s, time_gap, cond)
        C_s_n = self.get_c_s(time_gap, cond, num_grid=num_grid_s)
        intensity_s = q / C_s_n.unsqueeze(-1)
        if not with_survival:
            return tilde_lambda * intensity_s
        tilde_Lambda = self.get_tilde_Lambda(time_gap, cond, num_grid=num_grid_t)
        tilde_G_n = torch.exp(-tilde_Lambda)
        tilde_Lambda_T = self.get_tilde_Lambda_T(t, cond, num_grid=num_grid_t)
        tilde_G_n_T = torch.exp(-tilde_Lambda_T)
        F_n = torch.sigmoid(self.get_ending_logit(cond)[:,0]).reshape(-1,1,1)
        
        lambda_t = tilde_G_n * tilde_lambda / (tilde_G_n + ((1-tilde_G_n_T)/F_n) - 1)
        # make sure lambda_t is non-negative
        lambda_t = torch.where(lambda_t < 0, torch.zeros_like(lambda_t), lambda_t)


        
        intensity = lambda_t * intensity_s

        return intensity


    