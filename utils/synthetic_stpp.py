#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STPPG: Spatio-Temporal Point Process Generator

References:
- https://www.jstatsoft.org/article/view/v053i02
- https://www.ism.ac.jp/editsec/aism/pdf/044_1_0001.pdf
- https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator

Dependencies:
- Python 3.6.7
"""

import sys
import arrow
import numpy as np
from scipy.stats import norm


import numpy as np

import math

def _norm_cdf(x):
    """
    CDF of standard normal distribution, implemented using erf, supports numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / np.sqrt(2.0)))



def gaussian_box_mass(mu_x, mu_y, sigma_x, sigma_y, a, b, c, d):
    """
    Calculate the integral of 2D Gaussian N((mu_x,mu_y), diag(sigma_x^2,sigma_y^2))
    over the rectangle [a,b] x [c,d].
    mu_x, mu_y can be scalars or arrays of shape (n,).
    Return the mass of the same shape.
    """
    ax = (a - mu_x) / sigma_x
    bx = (b - mu_x) / sigma_x
    cy = (c - mu_y) / sigma_y
    dy = (d - mu_y) / sigma_y

    mass_x = _norm_cdf(bx) - _norm_cdf(ax)  # shape the same as mu_x
    mass_y = _norm_cdf(dy) - _norm_cdf(cy)  # shape the same as mu_y

    return mass_x * mass_y  # Broadcasted to the same shape


def gaussian_2d(s, mean, sigma_x, sigma_y):
    """
    2D Gaussian density with diagonal covariance.
    s:     (..., 2)
    mean:  (2,)
    """
    s = np.asarray(s)
    mean = np.asarray(mean)
    diff = s - mean  # (..., 2)
    dx = diff[..., 0] / sigma_x
    dy = diff[..., 1] / sigma_y
    norm_const = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    return norm_const * np.exp(-0.5 * (dx ** 2 + dy ** 2))




class AutoIntGaussianSelfCorrecting(object):
    """
    Self-correcting spatial point process kernel (Gaussian spatial kernel):

        λ*(s,t) = μ · exp( β t g0(s) - Σ_{i: t_i < t} α g2(s, s_i) )

    其中：
        g0(s)      = N2(s; center0, diag(sigma0_x^2, sigma0_y^2))  (raw, Gaussian pdf)
        g2(s,s_i)  = N2(s; s_i,    diag(sigma2_x^2, sigma2_y^2))   (raw, Gaussian pdf)
        g1(t,t_i)  = α · 1_{t > t_i}  (to compatible interface)

    注意：
      - Here we no longer use spatial integrals like λ_T(t)=∫_R λ*(u,t)du, only give local intensity λ*(s,t);
      - g0,g2 are still raw Gaussian pdfs that integrate to 1 over the entire R^2;
      - max_intensity gives a rough conservative upper bound on the time interval [0, T_max].
    """

    def __init__(self,
                 mu=0.5,
                 alpha=0.6,
                 beta=2.0,
                 sigma0_x=0.8,
                 sigma0_y=0.8,
                 sigma2_x=0.3,
                 sigma2_y=0.3,
                 center0=(0.0, 0.0),
                 S=[[0., 1.], [0., 1.]],
                 T_max=1.0):
        """
        参数：
          mu      : self-correcting baseline intensity μ (overall scale)
          alpha   : inhibition strength α for each historical point (on g2)
          beta    : time growth coefficient β
          sigma0_ : spatial scale of g0
          sigma2_ : spatial scale of g2
          center0 : center of g0 (m0_x, m0_y)
          S       : bounded region [a,b] x [c,d] (only saved, not used for calculation)
          T_max   : time upper bound for simulation/usage, used for max_intensity
        """
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.sigma0_x = float(sigma0_x)
        self.sigma0_y = float(sigma0_y)
        self.sigma2_x = float(sigma2_x)
        self.sigma2_y = float(sigma2_y)

        self.center0 = np.asarray(center0, dtype=float)

        # Region R = [a,b] x [c,d] (currently only saved, not used for calculation)
        self.S = S
        self.a, self.b = self.S[0]
        self.c, self.d = self.S[1]

        # Time interval upper bound (used for max_intensity)
        self.T_max = float(T_max)

    # ---------- Spatial kernel (raw Gaussian) ----------

    def g0(self, s):
        """Background spatial kernel g0(s), raw 2D Gaussian pdf."""
        return gaussian_2d(s, self.center0, self.sigma0_x, self.sigma0_y)

    def g2(self, s, his_s):
        """
        Triggering spatial kernel g2(s, s_i) for all historical points.
        his_s: shape (n,2) or (0,2), return array of shape (n,).
        """
        his_s = np.asarray(his_s)
        if his_s.size == 0:
            return np.zeros(0, dtype=float)
        # gaussian_2d supports broadcast, here x=his_s, mu=s
        return gaussian_2d(his_s, s, self.sigma2_x, self.sigma2_y)

    # ---------- Temporal "kernel" g1 (only to compatible interface) ----------

    def g1(self, t, his_t):
        """
        In self-correcting, use
            g1(t, t_i) = α · 1_{t > t_i}
        so that Σ g1(t,t_i) g2(s,s_i) = α Σ_{t_i<t} g2(s,s_i)
        can be put into the exponent.
        """
        his_t = np.asarray(his_t)
        if his_t.size == 0:
            return np.zeros(0, dtype=float)

        delta_t = t - his_t
        out = np.zeros_like(delta_t, dtype=float)
        mask = delta_t > 0
        out[mask] = self.alpha
        return out

    # ---------- Raw total intensity λ*(s,t) ----------

    def nu(self, t, s, his_t, his_s):
        """
        Self-correcting intensity:
            λ*(s,t) = μ · exp( β t g0(s) - Σ_i g1(t,t_i) g2(s,s_i) )

        where:
            g1(t,t_i) = α · 1_{t>t_i}
            g2(s,s_i) = 2D Gaussian pdf
        """
        g0_val = self.g0(s)

        his_t = np.asarray(his_t)
        his_s = np.asarray(his_s)

        if his_t.size == 0 or his_s.size == 0:
            exponent = self.beta * t * g0_val
        else:
            g1_vals = self.g1(t, his_t)   # (n,)
            g2_vals = self.g2(s, his_s)   # (n,)
            exponent = self.beta * t * g0_val - np.sum(g1_vals * g2_vals)

        return self.mu * np.exp(exponent)

    # ---------- Global approximate upper bound, used for thinning ----------

    def max_intensity(self, safety_factor=20.0, T_max=None):
        """
        Give a "global approximate upper bound" λ_max, used for Ogata thinning etc.

        For self-correcting form:
            λ*(s,t) = μ exp( β t g0(s) - α Σ g2(s,s_i) )

        Because -α Σ g2 ≤ 0, so the maximum value appears at: t = T_max and Σ g2 = 0, and g0(s) takes the sup.

        The sup of the Gaussian pdf is sup_s g0(s) = 1 / (2π σ0_x σ0_y).

        Therefore:
            λ_max ≈ μ · exp( β T_max · g0_max )

        Multiply by a safety_factor to slightly relax.
        """
        if T_max is None:
            T_max = self.T_max

        # Spatial part: sup of the Gaussian pdf
        g0_max = 1.0 / (2.0 * np.pi * self.sigma0_x * self.sigma0_y)

        exponent_max = self.beta * float(T_max) * g0_max
        lam_star_max = self.mu * np.exp(exponent_max)

        return safety_factor * lam_star_max


class AutoIntGaussianHawkes(object):
    """
    Raw Hawkes intensity:
        λ*(s,t) = μ g0(s) + Σ_{i: t_i < t} g1(t, t_i) g2(s, s_i),

    where:
        g1(t, t_i) = α exp(-β (t - t_i)) 1_{t > t_i},
        g0(s)      = N2(s; center0, diag(sigma0_x^2, sigma0_y^2))  (raw, Gaussian pdf)
        g2(s,s_i)  = N2(s; s_i,    diag(sigma2_x^2, sigma2_y^2))   (raw, Gaussian pdf)

    Defined on the bounded rectangle R = [a,b] x [c,d]:
        λ_T(t)      = ∫_R λ*(u,t) du
                     = μ Z0 + Σ_i g1(t,t_i) Z2(s_i),

        p_S(s | t, H_t) = λ*(s,t) / λ_T(t) · 1_{s∈R}.

    Note:
      - g0,g2 are raw Gaussian pdfs that integrate to 1 over the entire R^2, no truncation normalization;
      - λ_T(t) is the integral of λ*(u,t) over s∈R (“time edge intensity within the region”);
      - This expression for λ*(s,t) is exactly the same as the original Hawkes, except we additionally define λ_T and p_S.
    """

    def __init__(self,
                 mu=0.5,
                 alpha=0.6,
                 beta=2.0,
                 sigma0_x=0.8,
                 sigma0_y=0.8,
                 sigma2_x=0.3,
                 sigma2_y=0.3,
                 center0=(0.0, 0.0),
                 S=[[0., 1.], [0., 1.]]):
        """
        参数：
          mu      : time baseline μ
          alpha   : time Hawkes intensity α
          beta    : time decay β
          sigma0_ : spatial scale of g0
          sigma2_ : spatial scale of g2
          center0 : center of g0 (m0_x, m0_y)
          S      : bounded region [a,b] x [c,d]
        """
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.sigma0_x = float(sigma0_x)
        self.sigma0_y = float(sigma0_y)
        self.sigma2_x = float(sigma2_x)
        self.sigma2_y = float(sigma2_y)

        self.center0 = np.asarray(center0, dtype=float)

        # Region R = [a,b] x [c,d]
        self.S = S
        self.a, self.b = self.S[0]
        self.c, self.d = self.S[1]

        # Z0 is the integral of g0 over S (only depends on center0, sigma0, S, can be precomputed)
        self.Z0 = gaussian_box_mass(self.center0[0], self.center0[1],
                                    self.sigma0_x, self.sigma0_y,
                                    self.a, self.b, self.c, self.d)

    # ---------- Spatial kernel (raw Gaussian) ----------

    def g0(self, s):
        """Background spatial kernel g0(s), raw 2D Gaussian pdf."""
        return gaussian_2d(s, self.center0, self.sigma0_x, self.sigma0_y)

    def g2(self, s, his_s):
        """
        Triggering spatial kernel g2(s, s_i) for all historical points.
        his_s: shape (n,2) or (0,2), return array of shape (n,).
        """
        his_s = np.asarray(his_s)
        if his_s.size == 0:
            return np.zeros(0, dtype=float)
        # Note: gaussian_2d supports broadcast, here x=his_s, mu=s
        return gaussian_2d(his_s, s, self.sigma2_x, self.sigma2_y)

    # ---------- Temporal kernel g1 ----------

    def g1(self, t, his_t):
        """
        Temporal kernel:
            g1(t, t_i) = α exp(-β (t - t_i)) 1_{t > t_i}.
        Return array of shape (n,).
        """
        his_t = np.asarray(his_t)
        if his_t.size == 0:
            return np.zeros(0, dtype=float)

        delta_t = t - his_t
        out = np.zeros_like(delta_t, dtype=float)
        mask = delta_t > 0
        out[mask] = self.alpha * np.exp(-self.beta * delta_t[mask])
        return out

    # ---------- Raw total intensity λ*(s,t) (no normalization) ----------

    def nu(self, t, s, his_t, his_s):
        """
        Raw Hawkes intensity:
            λ*(s,t) = μ g0(s) + Σ_i g1(t,t_i) g2(s,s_i)
        where:
            g0(s) = raw Gaussian pdf that integrates to 1 over the entire R^2
            g2(s,s_i) = raw Gaussian pdf that integrates to 1 over the entire R^2
        """
        base = self.mu * self.g0(s)

        his_t = np.asarray(his_t)
        his_s = np.asarray(his_s)

        if his_t.size == 0 or his_s.size == 0:
            return base

        g1_vals = self.g1(t, his_t)   # (n,)
        g2_vals = self.g2(s, his_s)   # (n,)
        return base + np.sum(g1_vals * g2_vals)

    # ---------- Time edge intensity λ_T(t) after integrating over R ----------

    def lambda_T(self, t, his_t, his_s):
        """
        λ_T(t) = ∫_R λ*(u,t) du
               = μ Z0 + Σ_i g1(t,t_i) Z2(s_i),
        where:
          Z0      = ∫_R g0(u) du
          Z2(s_i) = ∫_R g2(u, s_i) du
                  = Gaussian N((x_i,y_i), diag(sigma2_x^2,sigma2_y^2))
                     on the rectangle R.
        """
        his_t = np.asarray(his_t)
        his_s = np.asarray(his_s)

        # Background term μ Z0
        lam = self.mu * self.Z0

        if his_t.size == 0 or his_s.size == 0:
            return lam

        # Temporal kernel g1(t,t_i)
        g1_vals = self.g1(t, his_t)  # (n,)

        # Z2(s_i) for each historical point
        xs = his_s[:, 0]
        ys = his_s[:, 1]
        Z2_all = gaussian_box_mass(xs, ys,
                                   self.sigma2_x, self.sigma2_y,
                                   self.a, self.b, self.c, self.d)  # (n,)

        lam += np.sum(g1_vals * Z2_all)
        return lam

    # ---------- Spatial conditional density p_S(s | t, H_t) ----------

    def p_S(self, t, s, his_t, his_s):
        """
            Spatial conditional density:
            p_S(s | t, H_t) = λ*(s,t) / λ_T(t) · 1_{s ∈ R}.
        """
        lam_star = self.lambda_star(t, s, his_t, his_s)
        lam_T = self.lambda_T(t, his_t, his_s)

        if lam_T <= 0.0:
            return 0.0

        # If s is not in R, here we can directly return 0 (by definition)
        x, y = float(s[0]), float(s[1])
        if not (self.a <= x <= self.b and self.c <= y <= self.d):
            return 0.0

        return lam_star / lam_T

    # ---------- Convenient method: factorized form of λ*(s,t) ----------

    def factorized_lambda(self, t, s, his_t, his_s):
        """
        Use factorized form to calculate λ*(s,t):
            λ*(s,t) = λ_T(t) * p_S(s | t, H_t)
        (Theoretically the same as lambda_star(t,s,...), used for sanity check.)
        """
        lam_T = self.lambda_T(t, his_t, his_s)
        p_s = self.p_S(t, s, his_t, his_s)
        return lam_T * p_s

    def max_intensity(self, safety_factor=2.0):
        """
        Give a "global approximate upper bound" λ_max, used for Ogata thinning etc.

        For the form:
          λ*(s,t) = μ g0(s) + Σ_i g1(t,t_i) g2(s,s_i)

        Spatial part:
          sup_s g0(s) = 1 / (2π σ0_x σ0_y)
          sup_s g2(s,s_i) = 1 / (2π σ2_x σ2_y)

        Time part:
          Treat μ + Σ_i g1(t,t_i) as one-dimensional Hawkes intensity,
          In α/β < 1, the steady-state intensity ≈ μ / (1 - α/β),
          therefore Σ_i g1(t,t_i) ≲ λ_1D_max - μ.

        Finally:
          λ*(s,t) ≤ μ g0_max + (λ_1D_max - μ) g2_max

        Note: This is a parameter-level conservative upper bound, not dependent on specific history.
        """

        # ---- Time part: coarse upper bound for one-dimensional Hawkes ----
        if self.beta > 0 and self.alpha < self.beta:
            branching_ratio = self.alpha / self.beta
            lam_1d_max = self.mu / max(1.0 - branching_ratio, 1e-6)
        else:
            # Unstable or boundary case, give a conservative linear upper bound
            lam_1d_max = self.mu * (1.0 + abs(self.alpha / max(self.beta, 1e-6)))

        # ---- Spatial part: sup of Gaussian pdf ----
        g0_max = 1.0 / (2.0 * np.pi * self.sigma0_x * self.sigma0_y)
        g2_max = 1.0 / (2.0 * np.pi * self.sigma2_x * self.sigma2_y)

        # Upper bound for Σ g1(t,t_i) (coarse): ≈ λ_1d_max - μ
        sum_g1_max = max(lam_1d_max - self.mu, 0.0)

        # Upper bound for all s,t intensities
        lam_star_max = self.mu * g0_max + sum_g1_max * g2_max

        return safety_factor * lam_star_max






class HawkesLam(object):
    """Intensity of Spatio-temporal Hawkes point process"""
    def __init__(self, mu, kernel, maximum=1e+4):
        self.mu      = mu
        self.kernel  = kernel
        self.maximum = maximum

    def value(self, t, his_t, s, his_s):
        """
        return the intensity value at (t, s).
        The last element of seq_t and seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """
        # if len(his_t) > 0:
        #     val = self.mu + np.sum(self.kernel.nu(t, s, his_t, his_s))
        # else:
        #     val = self.mu
        val = np.sum(self.kernel.nu(t, s, his_t, his_s))
        return val

    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Hawkes processes"

class SpatialTemporalPointProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, lam):
        """
        Params:
        """
        # model parameters
        self.lam     = lam

    def _homogeneous_poisson_sampling(self, T=[0, 1], S=[[0, 1], [0, 1]]):
        """
        To generate a homogeneous Poisson point pattern in space S X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S X T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.

        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            S:   [(min_t, max_t), (min_x, max_x), (min_y, max_y), ...] indicates the
                range of coordinates regarding a square (or cubic ...) region.
        Returns:
            samples: point process samples:
            [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
        """
        _S     = [T] + S
        # sample the number of events from S
        n      = utils.lebesgue_measure(_S)
        N      = np.random.poisson(size=1, lam=self.lam.upper_bound() * n)
        # simulate spatial sequence and temporal sequence separately.
        points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
        points = np.array(points).transpose()
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points[:, 0].argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape), file=sys.stderr)
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            t     = homo_points[i, 0]
            s     = homo_points[i, 1:]
            his_t = retained_points[:, 0]
            his_s = retained_points[:, 1:]
            # thinning
            lam_value = self.lam.value(t, his_t, s, his_s)
            lam_bar   = self.lam.upper_bound()
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("intensity %f is greater than upper bound %f." % (lam_value, lam_bar), file=sys.stderr)
                return None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, homo_points[[i], :]], axis=0)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]), file=sys.stderr)
        # log the final results of the thinning algorithm
        if verbose:
            print("[%s] thining samples %s based on %s." % \
                (arrow.now(), retained_points.shape, self.lam), file=sys.stderr)
        return retained_points

    def generate(self, T=[0, 1], S=[[0, 1], [0, 1]], batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling(T, S)
            points      = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            print("[%s] %d-th sequence is generated." % (arrow.now(), b+1), file=sys.stderr)
            b += 1
        # fit the data into a tensor
        return points_list, sizes
        # data = np.zeros((batch_size, max_len, 3))
        # for b in range(batch_size):
        #     data[b, :points_list[b].shape[0]] = points_list[b]
        # return data, sizes





if __name__ == "__main__":

    seed = 2
    np.random.seed(seed)




    S = [[0., 1.], [0., 1.]]
    kernel = AutoIntGaussianSelfCorrecting(
        mu=1.,
        alpha=0.4,
        beta=0.2,
        sigma0_x=0.25,
        sigma0_y=0.25,
        sigma2_x=0.2,
        sigma2_y=0.2,
        center0=(0.5, 0.5),
        S=S
    )



    lam = HawkesLam(
        mu=None,              
        kernel=kernel,
        maximum=kernel.max_intensity()         
    )

    pp = SpatialTemporalPointProcess(lam)

    max_time = 40                 
    
    batch_size = 8000


    # generate points

    points, sizes = pp.generate(
            T=[0., max_time], S=S, 
            batch_size=batch_size, verbose=True)

    # print(points)
    print(sizes)
    # split the points into train, val and test
    train_points = points[:int(batch_size * 0.8)]
    val_points = points[int(batch_size * 0.8):int(batch_size * 0.9)]
    test_points = points[int(batch_size * 0.9):]

    # # read or save to local npy file.
    # save to pkl file, each element is a list of [t, t_diff, x, y]
    import pickle


    with open('/root/Desktop/code/WSM_STPP/dataset/SC1/data_train.pkl', 'wb') as f:
        pickle.dump(train_points, f)
    with open('/root/Desktop/code/WSM_STPP/dataset/SC1/data_val.pkl', 'wb') as f:
        pickle.dump(val_points, f)
    with open('/root/Desktop/code/WSM_STPP/dataset/SC1/data_test.pkl', 'wb') as f:
        pickle.dump(test_points, f)
