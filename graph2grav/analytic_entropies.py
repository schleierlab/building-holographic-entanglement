# Adapted from the Mathematica notebooks of Datta and David (2014), but with corrections from Chen and Wu (2015)
# The bulk of the adaptation was performed by ChatGPT 5.1 Thinking.

import numpy as np
import mpmath as mp
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm


def periodic_CFT_entropy(length, total_length, central_charge=1, offset=0):
    """
    Returns the entanglement entropy for a contiguous region of given length of a periodic 1+1D CFT.
    `total_length` is the length of the entire system, and `central_charge` is the central charge of the CFT.
    """

    return (central_charge / 3) * np.log(np.sin(np.pi * length / total_length)) + offset

#############

# ----------------------------------------------------------------------
# Jacobi theta function and its derivative (Mathematica EllipticTheta[1])
# mpmath.jtheta implements Jacobi theta functions and supports derivatives
#   jtheta(n, z, q, derivative=d)   :contentReference[oaicite:0]{index=0}
# ----------------------------------------------------------------------

mp.mp.dps = 25  # precision; maybe lower for speed

# ----------------------------------------------------------------------
# Theta functions (same as before)
# ----------------------------------------------------------------------

def q(beta):
    """Elliptic nome q = exp(-π β) for Jacobi theta functions."""
    return mp.e**(-mp.pi * beta)  # tau = i beta

def theta1(z, beta):
    """Jacobi theta function θ₁(z|τ) with τ = i β."""
    return mp.jtheta(1, z, q(beta))

def theta1_prime(z, beta):
    """Derivative ∂θ₁/∂z of Jacobi theta function."""
    return mp.jtheta(1, z, q(beta), 1)

def theta1_bar(zbar, beta):
    """
    \bar{\vartheta}_1(\bar z | tau) for purely imaginary tau, 
    implemented as complex conjugate.
    """
    # z = conj(zbar); for our paths zbar is simple enough that this is fine
    return mp.conj(theta1(mp.conj(zbar), beta))

# ----------------------------------------------------------------------
# W-matrix entries W_a^{b(k)} (Chen & Wu eqs. in your message)
# z1 = y, z2 = 1 - y, tau = i beta
# ----------------------------------------------------------------------

def W11(k, n, y, beta):
    """W-matrix element W₁¹⁽ᵏ⁾ from Chen & Wu (2015) for CFT entanglement.

    Parameters
    ----------
    k : int
        Replica index (0 ≤ k < n).
    n : int
        Replica number.
    y : float
        Position parameter.
    beta : float
        Inverse temperature.

    Returns
    -------
    mpmath number
        W-matrix element.
    """
    alpha = k / n
    z1 = mp.mpf(y)
    z2 = mp.mpf(1.0 - y)

    def integrand(z):
        z = mp.mpf(z)
        a = theta1(mp.pi*(z - z1), beta)**(-(1 - alpha))
        b = theta1(mp.pi*(z - z2), beta)**(-alpha)
        c = theta1(mp.pi*(z - ((1 - alpha)*z1 + alpha*z2)), beta)
        return a * b * c

    # split at z1 and z2 (branch points)
    return mp.quad(integrand, [0, z1, z2, 1])


def W12(k, n, y, beta):
    """W-matrix element W₁²⁽ᵏ⁾ from Chen & Wu (2015) for CFT entanglement.

    Parameters
    ----------
    k : int
        Replica index (0 ≤ k < n).
    n : int
        Replica number.
    y : float
        Position parameter.
    beta : float
        Inverse temperature.

    Returns
    -------
    mpmath number
        W-matrix element.
    """
    alpha = k / n
    z1 = mp.mpf(y)
    z2 = mp.mpf(1.0 - y)

    def integrand(zbar):
        zbar = mp.mpf(zbar)
        a = theta1_bar(mp.pi*(zbar - z1), beta)**(-alpha)
        b = theta1_bar(mp.pi*(zbar - z2), beta)**(-(1 - alpha))
        c = theta1_bar(mp.pi*(zbar - (alpha*z1 + (1 - alpha)*z2)), beta)
        return a * b * c

    return mp.quad(integrand, [0, z1, z2, 1])

def W21(k, n, y, beta):
    """
    W_2^{1(k)} = ∫_0^τ dz θ1(z-z1)^{-(1-k/n)} θ1(z-z2)^(-k/n)
                        θ1(z-(1-k/n)z1 - (k/n)z2)
    where τ = i β. Parametrize z = i t, t∈[0,β], so dz = i dt.
    """
    alpha = k / n
    z1 = y
    z2 = 1.0 - y
    beta_mp = mp.mpf(beta)

    def integrand(t):
        t = mp.mpf(t)
        z = 1j * t
        a = theta1(mp.pi*(z - z1), beta)**(-(1 - alpha))
        b = theta1(mp.pi*(z - z2), beta)**(-alpha)
        c = theta1(mp.pi*(z - ((1 - alpha)*z1 + alpha*z2)), beta)
        return a * b * c * 1j  # dz/dt = i

    return mp.quad(integrand, [0, beta_mp])

def W22(k, n, y, beta):
    """
    W_2^{2(k)} = ∫_0^{\bar τ} d\bar z \bar θ1(\bar z- \bar z1)^(-k/n)
                               \bar θ1(\bar z- \bar z2)^(-(1-k/n))
                               \bar θ1(\bar z - (k/n) z1 - (1-k/n) z2)
    with \bar τ = -i β. Parametrize \bar z = -i t, t∈[0,β], so d\bar z = -i dt.
    """
    alpha = k / n
    z1 = y
    z2 = 1.0 - y
    beta_mp = mp.mpf(beta)

    def integrand(t):
        t = mp.mpf(t)
        zbar = -1j * t
        a = theta1_bar(mp.pi*(zbar - z1), beta)**(-alpha)
        b = theta1_bar(mp.pi*(zbar - z2), beta)**(-(1 - alpha))
        c = theta1_bar(mp.pi*(zbar - (alpha*z1 + (1 - alpha)*z2)), beta)
        return a * b * c * (-1j)  # d\bar z/dt = -i

    return mp.quad(integrand, [0, beta_mp])

# ----------------------------------------------------------------------
# Correct determinant and S_n
# ----------------------------------------------------------------------

def detW(k, n, y, beta):
    """
    det W^{(k)} = W_1^{1(k)} W_2^{2(k)} - W_1^{2(k)} W_2^{1(k)}
    (Chen & Wu’s matrix structure).
    """
    w11 = W11(k, n, y, beta)
    w12 = W12(k, n, y, beta)
    w21 = W21(k, n, y, beta)
    w22 = W22(k, n, y, beta)
    return w11 * w22 - w12 * w21

def S2(n, y, beta):
    """
    S_n using corrected detW (Chen–Wu), with mpmath complex arithmetic.
    """
    prod = mp.mpc(1)  # start as mpmath complex
    for k in range(n):
        prod *= detW(k, n, y, beta)

    n_mp = mp.mpf(n)
    beta_mp = mp.mpf(beta)

    # use mpmath.log, NOT numpy.log
    val = (-mp.log(prod) + n_mp * mp.log(2 * beta_mp)) / (1 - n_mp)

    return float(mp.re(val))

def SE1(y, beta):
    """
    Your analytic n→1 piece (unchanged).
    """
    z = mp.pi * (1 - 2*y)
    num = theta1(z, beta)
    den = theta1_prime(0.0, beta)
    return (2.0/3.0) * float(mp.re(mp.log(abs(num / den))))

def build_interpolator(beta,
                       n_min=2, n_max=5,
                       y_min=0.00001, y_max=0.4999999,
                       ny=200):
    """Build 2D interpolator for S₂(n,y) on grid at fixed β.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    n_min, n_max : int
        Replica number range.
    y_min, y_max : float
        Position parameter range.
    ny : int
        Number of y grid points.

    Returns
    -------
    interp : RegularGridInterpolator
        Interpolator for S₂(n,y).
    n_vals : ndarray
        Replica number grid.
    y_vals : ndarray
        Position parameter grid.
    S2_vals : ndarray
        S₂ values on grid.
    """
    n_vals = np.arange(n_min, n_max+1, dtype=float)
    y_vals = np.linspace(y_min, y_max, ny)

    S2_vals = np.empty((len(n_vals), len(y_vals)), dtype=float)

    # outer loop progress bar
    for i, n in enumerate(tqdm(n_vals, desc=f"β={beta} n-loop")):
        # inner loop progress bar (optional; remove if too noisy)
        for j, y in enumerate(tqdm(y_vals,
                                   desc=f"β={beta} y-loop (n={n})",
                                   leave=False)):
            S2_vals[i, j] = S2(int(n), float(y), beta)

    interp = RegularGridInterpolator(
        (n_vals, y_vals), S2_vals,
        bounds_error=False, fill_value=None
    )
    return interp, n_vals, y_vals, S2_vals


# ----------------------------------------------------------------------
# Example: reproduce the n→1 entanglement-entropy curves for β=0.3,0.5,0.7,0.9
# ----------------------------------------------------------------------

def entanglement_plot(betas=None):
    """Plot entanglement entropy S_E(L) for different β values.

    Parameters
    ----------
    betas : list of float, optional
        Inverse temperatures. Default [0.3].

    Returns
    -------
    betas : list
        Beta values used.
    L : ndarray
        Interval sizes.
    all_S_ent : list of ndarray
        Entropy curves for each beta.
    """
    if betas is None:
        betas = [0.3] # [0.3, 0.5, 0.7, 0.9]

    interps = {}

    for beta in betas:
        print(f"Building interpolator for beta={beta} (this is slow)...")
        f_beta, n_vals, y_vals, _ = build_interpolator(beta)
        interps[beta] = f_beta

    L = np.linspace(0.0, 1.0, 200)
    y = (1.0 - L) / 2.0

    plt.figure()

    all_S_ent = []

    for beta, color in zip(betas, ['r', 'b', 'g', 'm']):
        f_beta = interps[beta]
        # Evaluate interpolated S2 at n=1 (extrapolation in n)
        points = np.column_stack((np.ones_like(y), y))
        S2_at_1 = f_beta(points)

        SE1_vals = np.array([SE1(yy, beta) for yy in y])
        S_ent = S2_at_1 + SE1_vals

        all_S_ent.append(S_ent)

        plt.plot(L, S_ent, color=color, label=fr"$\beta={beta}$")

    plt.xlim(-0.01, 1.01)
    plt.ylim(-2.3, 5.0)
    plt.xlabel("L")
    plt.ylabel(r"$S_E$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return betas, L, all_S_ent

# ----------------------------------------------------------------------
# worker for multiprocessing
# ----------------------------------------------------------------------

def _S2_worker(args):
    """
    Single grid point worker: computes S2(n, y, beta) in its own process.
    We set mp.mp.dps inside the worker so each process has the right precision.
    """
    n, y, beta, dps = args
    mp.mp.dps = dps
    return S2(n, y, beta)


# ----------------------------------------------------------------------
# parallel grid builder
# ----------------------------------------------------------------------

def build_interpolator_mp(beta,
                          n_min=2, n_max=5,
                          y_min=0.00001, y_max=0.4999999,
                          ny=200,
                          dps=None,
                          max_workers=None):
    """
    Build S2(n,y,beta) grid and interpolation using multiprocessing.

    Returns: interp, n_vals, y_vals, S2_vals
    """
    if dps is None:
        dps = mp.mp.dps  # use current global precision by default

    n_vals = np.arange(n_min, n_max+1, dtype=float)
    y_vals = np.linspace(y_min, y_max, ny)

    # flatten tasks: each task is one (n, y)
    tasks = [(int(n), float(y), float(beta), dps)
             for n in n_vals for y in y_vals]

    S2_flat = np.empty(len(tasks), dtype=float)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for idx, val in enumerate(
                tqdm(ex.map(_S2_worker, tasks),
                     total=len(tasks),
                     desc=f"β={beta} S2 grid (mp)")):
            S2_flat[idx] = val

    S2_vals = S2_flat.reshape(len(n_vals), len(y_vals))

    interp = RegularGridInterpolator(
        (n_vals, y_vals), S2_vals,
        bounds_error=False, fill_value=None
    )
    return interp, n_vals, y_vals, S2_vals


# ----------------------------------------------------------------------
# entanglement entropy curves, using multiprocessing grid build
# ----------------------------------------------------------------------

def entanglement_plot_mp(betas=None,
                      n_min=2, n_max=5,
                      y_min=0.00001, y_max=0.4999999,
                      ny=200,
                      max_workers=None,
                      make_plot=True):
    """
    Compute n→1 entanglement entropy curves for given list of betas.

    Returns
    -------
    betas : list of float
    L     : 1D np.array of interval sizes in [0,1]
    all_S_ent : list of np.array, one S_E(L) curve per beta
    """
    if betas is None:
        betas = [0.3]

    interps = {}

    for beta in betas:
        print(f"Building interpolator for beta={beta} (this is slow)...")
        f_beta, n_vals, y_vals, _ = build_interpolator_mp(
            beta,
            n_min=n_min, n_max=n_max,
            y_min=y_min, y_max=y_max,
            ny=ny,
            dps=mp.mp.dps,
            max_workers=max_workers
        )
        interps[beta] = f_beta

    L = np.linspace(0.0, 1.0, 200)
    y = (1.0 - L) / 2.0

    all_S_ent = []

    if make_plot:
        plt.figure()

    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']

    for idx, beta in enumerate(betas):
        color = colors[idx % len(colors)]
        f_beta = interps[beta]

        # Evaluate interpolated S2 at n=1 (extrapolation in n)
        points = np.column_stack((np.ones_like(y), y))
        S2_at_1 = f_beta(points)

        SE1_vals = np.array([SE1(yy, beta) for yy in y])
        S_ent = S2_at_1 + SE1_vals
        all_S_ent.append(S_ent)

        if make_plot:
            plt.plot(L, S_ent, color=color, label=fr"$\beta={beta}$")

    if make_plot:
        plt.xlim(-0.01, 1.01)
        plt.ylim(-2.3, 5.0)
        plt.xlabel("Region Size")
        plt.ylabel(r"Entanglement Entropy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return betas, L, all_S_ent

def S1_renyi(n, y, beta):
    """
    S_1 piece in Datta–David eq. (3.4), specialized to z1=y, z2=1-y.
    This is the analytic 'universal' part, independent of the W-matrix.
    """
    n_mp = mp.mpf(n)
    z = mp.pi * (1 - 2.0*y)
    ratio = theta1(z, beta) / theta1_prime(0.0, beta)
    # ratio is complex; we want log|ratio|
    logabs = mp.log(mp.fabs(ratio))
    return float((n_mp + 1) / (3 * n_mp) * logabs)

def S_total(n, y, beta):
    """Total Rényi entropy S_n = S1 + S2 (theta and W contributions)."""
    return S1_renyi(n, y, beta) + S2(n, y, beta)  # S2 is your W-piece

def _S_total_worker(args):
    """
    Worker for full Rényi S_n (S1 + S2).
    """
    n, y, beta, dps = args
    mp.mp.dps = dps
    return S_total(n, y, beta)

def renyi2_plot(betas=None,
                y_min=1e-5, y_max=0.4999999,
                ny=200,
                make_plot=True):
    """
    Compute and (optionally) plot the n=2 Rényi entropy S_2(L)
    for a list of betas, using the existing S2(n,y,beta).

    Returns
    -------
    betas      : list of float
    L          : 1D np.array of interval sizes in [0,1]
    all_S2     : list of np.array, one S_2(L) curve per beta
    """
    if betas is None:
        betas = [0.3]

    # y runs from almost 0 to almost 1/2, just like before
    y_vals = np.linspace(y_min, y_max, ny)
    L = 1.0 - 2.0 * y_vals

    all_S2 = []

    if make_plot:
        plt.figure()
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']

    for idx, beta in enumerate(betas):
        color = colors[idx % len(colors)]
        S2_vals = np.empty_like(y_vals, dtype=float)

        for j, y in enumerate(tqdm(y_vals,
                                   desc=f"S2, β={beta}",
                                   leave=False)):
            S2_vals[j] = S_total(2, float(y), beta)

        all_S2.append(S2_vals)

        if make_plot:
            plt.plot(L, S2_vals, color=color, label=fr"$\beta={beta}$")

    if make_plot:
        plt.xlim(-0.01, 1.01)
        plt.xlabel("Interval length $L$")
        plt.ylabel(r"Rényi entropy $S_2$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return betas, L, all_S2

def renyi2_plot_mp(betas=None,
                   y_min=1e-5, y_max=0.4999999,
                   ny=200,
                   max_workers=None,
                   make_plot=True):
    """
    Compute and (optionally) plot the n=2 Rényi entropy S_2(L)
    for a list of betas, using multiprocessing.

    Returns
    -------
    betas_sorted : 1D np.array of β values (ascending)
    L_sorted     : 1D np.array of interval sizes in [0,1] (ascending)
    all_S2       : list of np.array, one S_2(L_sorted) curve per β
    S2_interp    : RegularGridInterpolator over (β, L) -> S_2
    """
    if betas is None:
        betas = [0.3]

    betas = list(betas)
    nb = len(betas)

    # y grid and corresponding L grid
    y_vals = np.linspace(y_min, y_max, ny)
    L_vals = 1.0 - 2.0 * y_vals

    # we’ll fill this as [beta_index, L_index]
    S2_grid = np.empty((nb, ny), dtype=float)

    dps = mp.mp.dps  # precision passed to workers

    if make_plot:
        plt.figure()
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']

    # run all betas in a single pool
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for ib, beta in enumerate(betas):
            # tasks for this beta: (n=2, y, beta, dps)
            tasks = [(2, float(y), float(beta), dps) for y in y_vals]

            row = np.empty_like(y_vals, dtype=float)
            for j, val in enumerate(
                    tqdm(ex.map(_S_total_worker, tasks),
                         total=len(tasks),
                         desc=f"S2, β={beta} (mp)",
                         leave=False)):
                row[j] = val

            S2_grid[ib, :] = row

    # sort L axis ascending (RegularGridInterpolator requirement)
    L_sorted_idx = np.argsort(L_vals)
    L_sorted = L_vals[L_sorted_idx]
    S2_grid = S2_grid[:, L_sorted_idx]

    # sort beta axis ascending as well
    beta_arr = np.array(betas, dtype=float)
    beta_sorted_idx = np.argsort(beta_arr)
    betas_sorted = beta_arr[beta_sorted_idx]
    S2_grid = S2_grid[beta_sorted_idx, :]

    # build list of curves (in the same order as betas_sorted)
    all_S2 = [S2_grid[i, :].copy() for i in range(nb)]

    # build interpolator over (β, L)
    S2_interp = RegularGridInterpolator(
        (betas_sorted, L_sorted),
        S2_grid,
        bounds_error=False,
        fill_value=None
    )

    if make_plot:
        for i, beta in enumerate(betas_sorted):
            color = colors[i % len(colors)]
            plt.plot(L_sorted, S2_grid[i, :],
                     color=color, label=fr"$\beta={beta}$")

        plt.xlim(-0.01, 1.01)
        plt.xlabel("Interval length $L$")
        plt.ylabel(r"Rényi entropy $S_2$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return betas_sorted, L_sorted, all_S2, S2_interp

if __name__ == "__main__":
    entanglement_plot()
