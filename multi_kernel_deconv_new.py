"""
Multi-kernel EMG deconvolution (IRLS + LSMR, convolution operators)

Compact summary:
- Estimate MUAP kernel widths from EMG PSD and construct derivative-of-Gaussian kernels.
- Recover sparse firing patterns with IRLS applied to the augmented system B = [A; sqrt(alpha) I],
  using weights W_i = 1 / sqrt(|r_i| + eps) built from the augmented residual r = Bx - c.
- Solve weighted least-squares with a LinearOperator and scipy.sparse.linalg.lsmr for memory efficiency.
- Regularization alpha is set by a heuristic alpha = 0.01 * lambda_max(A^T A).
- Includes numerical safeguards (adaptive eps, weight clipping, optional warm start).
"""

import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import welch, convolve
from scipy.sparse.linalg import lsmr, LinearOperator

# ------------ helper IO utilities ---------------
def _safe_to_array(val):
    """Try to convert to numpy array suitable for np.savez (reject object-dtype)."""
    try:
        arr = np.asarray(val)
        if arr.dtype == object:
            return False, None
        return True, arr
    except Exception:
        return False, None

def _clean_label(s):
    """Make a safe label for npz keys (no spaces, limited chars)."""
    if not isinstance(s, str):
        s = str(s)
    # simple cleanup: replace spaces and slashes
    return s.replace(' ', '_').replace('/', '_')

def save_results_list_to_npz(results, npz_path, meta=None, channel_names=None, compress=True):
    """
    Save `results` (list of per-channel dicts) into a compressed .npz file.
    channel_names: optional list of strings to map results[0] -> channel_names[0], etc.
    If a name is not provided for an index, falls back to 'ch{i}'.
    """
    npz_dict = {}

    for i, ent in enumerate(results):
        label = (channel_names[i] if (channel_names is not None and i < len(channel_names)) else f'ch{i}')
        label = _clean_label(label)
        prefix = f'{label}_'
        for key, val in ent.items():
            ok, arr = _safe_to_array(val)
            if ok:
                npz_dict[prefix + key] = arr
                continue

            # special-case recon_components: list of arrays possibly stackable
            if key == 'recon_components' and isinstance(val, (list, tuple)):
                try:
                    stacked = np.stack([np.asarray(v) for v in val], axis=0)
                    if stacked.dtype != object:
                        npz_dict[prefix + 'recon_components'] = stacked
                        continue
                except Exception:
                    pass
                for k_comp, comp in enumerate(val):
                    okc, arrc = _safe_to_array(comp)
                    if okc:
                        npz_dict[f'{prefix}recon_k{k_comp}'] = arrc
                    else:
                        npz_dict[f'{prefix}recon_k{k_comp}_txt'] = np.array([repr(comp)], dtype='U')
                continue

            # conv_history often list-of-floats -> try to save as float array
            if key == 'conv_history' and isinstance(val, (list, tuple)):
                try:
                    arr_ch = np.asarray(val, dtype=float)
                    npz_dict[prefix + key] = arr_ch
                    continue
                except Exception:
                    pass

            # last resort: JSON-serialize
            try:
                s = json.dumps(val, default=lambda o: str(o))
                npz_dict[prefix + key + '_json'] = np.array([s], dtype='U')
            except Exception:
                npz_dict[prefix + key + '_txt'] = np.array([repr(val)], dtype='U')

    # add channel count and meta
    npz_dict['__n_channels__'] = np.array([len(results)], dtype=np.int32)
    if meta is not None:
        try:
            npz_dict['__global_meta_json__'] = np.array([json.dumps(meta, default=str)], dtype='U')
        except Exception:
            npz_dict['__global_meta_txt__'] = np.array([repr(meta)], dtype='U')

    # ensure dir exists
    folder = os.path.dirname(npz_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    if compress:
        np.savez_compressed(npz_path, **npz_dict)
    else:
        np.savez(npz_path, **npz_dict)

# ------------ end IO utilities ---------------

def estimate_sigmas_from_gamma(signal, fs, mode):
    """
    Estimate Gaussian kernel sigmas from EMG PSD.
    mode: 'multi' -> linear fit on gamma vs f^2 -> symmetric two kernels
          'single' -> quadratic fit, return three percentile-based sigmas
    """
    nperseg = min(512, max(128, len(signal)))
    noverlap = nperseg // 2
    nfft = max(256, 2**int(np.ceil(np.log2(nperseg))))

    f, pxx = welch(signal, fs=fs, window='hamming',
                   nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=0)

    mask = f > 0
    f = f[mask]; pxx = pxx[mask]
    if f.size == 0:
        warnings.warn("No positive frequency bins from welch; returning default sigmas.")
        return [0.003, 0.003] if mode == 'multi' else [0.003, 0.003, 0.003]

    f_med = np.median(f); f_std = np.std(f)
    mask2 = (f >= (f_med - f_std)) & (f <= (f_med + 2*f_std))
    f_sel = f[mask2]; pxx_sel = pxx[mask2]

    if f_sel.size < 3:
        warnings.warn('Selected PSD band too small — falling back to full positive frequency band for sigma estimation.', UserWarning)
        f_sel = f; pxx_sel = pxx

    if f_sel.size < 3:
        warnings.warn('PSD too short to estimate sigma reliably — using default sigma=0.003s', UserWarning)
        return [0.003, 0.003] if mode == 'multi' else [0.003, 0.003, 0.003]

    # gamma = log( PSD / (4*pi^2 f^2) )
    eps = np.finfo(float).eps
    gamma = np.log(pxx_sel / (4 * np.pi**2 * f_sel**2 + eps))
    f2 = f_sel**2

    if mode == 'multi':
        p = np.polyfit(f2, gamma, 1)
        slope = p[0]
        if slope >= 0:
            warnings.warn("slope >= 0 from gamma vs f^2 fit — check PSD or selection band.")
        sigma2 = -slope / (4 * np.pi**2)
        sigma = np.sqrt(max(sigma2, eps))
        print("DEBUG sigmas:", sigma)
        return np.array([sigma, sigma])
    else:
        p = np.polyfit(f2, gamma, 2)  # a f^4 + b f^2 + c
        a, b = p[0], p[1]
        pctls = [0.15, 0.5, 0.85]
        sigmas = []

        # PSD-weighted percentiles (more robust + interpolation)
        pxx_sum = np.sum(pxx_sel)
        if pxx_sum <= 0 or not np.isfinite(pxx_sum):
            # fallback to uniform weighting if PSD is degenerate
            pxx_norm = np.ones_like(pxx_sel) / len(pxx_sel)
        else:
            # ensure non-negative, finite
            pxx_sel_pos = np.maximum(pxx_sel, 0.0)
            pxx_norm = pxx_sel_pos / (np.sum(pxx_sel_pos) + 1e-20)

        cdf = np.cumsum(pxx_norm)
        # ensure strictly increasing cdf for interp (last element should be 1)
        cdf[-1] = 1.0

        for pc in pctls:
            # interpolate percentile to get a smooth f0 (avoid discrete index jumps)
            f0 = np.interp(pc, cdf, f_sel)

            # compute derivative of polyfit (gamma = a*(f^2)^2 + b*(f^2) + c)
            # derivative wrt (f^2) is: dgamma/d(f^2) = 2*a*(f^2) + b
            f0_sq = f0**2
            slope = 2.0 * a * f0_sq + b

            if slope >= 0:
                warnings.warn(f"slope >= 0 from gamma fit at PSD-weighted percentile {pc:.2f} — using fallback linear fit.")
                # fallback: do linear fit on full f2 vs gamma (like 'multi' mode) and evaluate at f0
                p_lin = np.polyfit(f2, gamma, 1)
                slope = p_lin[0]

            sigma2 = -slope / (4.0 * np.pi**2)
            if not np.isfinite(sigma2) or sigma2 <= 0:
                # final safe fallback
                sigma = 0.003
            else:
                sigma = np.sqrt(sigma2)
            sigmas.append(sigma)

        print("DEBUG sigmas:", sigmas)
        return np.array(sigmas)
        
def build_kernels(sigmas, fs, mode):
    """
    Build first-derivative Gaussian kernels.
    Kernel length is chosen to ensure odd length and center at t=0.
    """
    tmax = 0.025
    nT = int(round(2 * tmax * fs)) + 1   # +1 to ensure odd length and center sample at 0
    t = np.linspace(-tmax, tmax, nT)
    nK = len(sigmas)
    K = np.zeros((nT, nK))
    for k, sigma in enumerate(sigmas):
        kernel = -t / (sigma**3 * np.sqrt(2*np.pi)) * np.exp(-t**2 / (2*sigma**2))
        if mode == 'multi' and k == 1:
            kernel = -kernel
        # store raw kernel (no normalization here)
        K[:, k] = kernel
    return K

def estimate_lam_max(apply_A, apply_AT, n, n_iters=8):
    """
    Power-iteration estimate of the largest eigenvalue of A^T A.
    Returns a positive real scalar (raises if non-positive).
    """
    v = np.random.randn(n)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(n_iters):
        w = apply_AT(apply_A(v))
        v = w / (np.linalg.norm(w) + 1e-12)
    # final Rayleigh quotient
    w = apply_AT(apply_A(v))
    lam = np.dot(v, w) / (np.dot(v, v) + 1e-12)

    # ensure a real non-negative value; raise on clear negative result
    lam = np.real(lam)
    if lam <= 0:
        raise ValueError("Estimated largest eigenvalue is non-positive — check apply_A/apply_AT.")
    return lam

def adjoint_test(apply_A, apply_AT, n, N, n_trials=5, tol=1e-8, rng_seed=1234, verbose=True):
    """
    Verify adjointness: check <A x, y> == <x, A^T y> for random tests.
    Returns list of results with absolute and relative errors.
    """
    rng = np.random.default_rng(rng_seed)
    results = []
    eps = 1e-16

    for t in range(n_trials):
        x = rng.standard_normal(n)
        y = rng.standard_normal(N)

        Ax = apply_A(x)        # length N
        ATy = apply_AT(y)      # length n

        # use complex-conjugate inner product in case of complex values
        lhs = np.vdot(Ax, y)
        rhs = np.vdot(x, ATy)

        # reduce to real part (theoretical result should be real)
        lhs_r = np.real(lhs)
        rhs_r = np.real(rhs)

        abs_err = float(np.abs(lhs_r - rhs_r))
        denom = max(abs(lhs_r), abs(rhs_r), eps)
        rel_err = abs_err / denom

        ok = rel_err < tol
        results.append({
            'trial': t,
            'lhs': lhs_r,
            'rhs': rhs_r,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'ok': ok
        })

        if verbose:
            print(f"[trial {t}] lhs={lhs_r:.6e}, rhs={rhs_r:.6e}, abs_err={abs_err:.3e}, rel_err={rel_err:.3e}, ok={ok}")

    n_ok = sum(1 for r in results if r['ok'])
    if verbose:
        print(f"Adjoint test: {n_ok}/{n_trials} trials passed (tol={tol})")

    return results

def resample_to_frames(X_mat, fs, frame_rate=30.0, method='mean'):
    """
    Resample per-sample coefficient matrix X_mat into per-frame values.

    Parameters
    ----------
    X_mat : array_like, shape (N, nK) or (N,)
        Deconvolution coefficients per original sample.
    fs : float
        Original sampling frequency (samples per second).
    frame_rate : float, optional
        Target frame rate (frames per second). Default is 30.0 (1 frame = 1/30 s).
    method : {'mean', 'sum', 'nearest'}, optional
        Aggregation method used to compute one value per frame:
        - 'mean'    : average of samples inside the frame window (default)
        - 'sum'     : sum of samples inside the frame window
        - 'nearest' : take the sample nearest to the frame center

    Returns
    -------
    X_frame : ndarray, shape (n_frames, nK)
        Aggregated coefficients per frame.
    frame_centers : ndarray, shape (n_frames,)
        Time (seconds) of each frame center.
    frame_count : int
        Number of frames (n_frames).
    """
    # Ensure numpy array and 2D shape (N, nK)
    X_mat = np.asarray(X_mat)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)
    N, nK = X_mat.shape

    # Compute total frames
    total_seconds = N / float(fs)
    n_frames = int(np.ceil(total_seconds * frame_rate))

    # sample times in seconds
    sample_times = np.arange(N) / float(fs)

    # frame edges (seconds): length n_frames+1
    frame_edges = np.arange(0, n_frames + 1) / frame_rate

    # preallocate output
    X_frame = np.zeros((n_frames, nK), dtype=float)

    if method in ('mean', 'sum'):
        # Use searchsorted to find indices of samples within each frame window
        starts = np.searchsorted(sample_times, frame_edges[:-1], side='left')
        ends   = np.searchsorted(sample_times, frame_edges[1:], side='left')

        for j in range(n_frames):
            s_idx = starts[j]
            e_idx = ends[j]
            if e_idx > s_idx:
                block = X_mat[s_idx:e_idx, :]
                if method == 'mean':
                    X_frame[j, :] = block.mean(axis=0)
                else:  # method == 'sum'
                    X_frame[j, :] = block.sum(axis=0)
            else:
                # no samples in this frame => leave zeros
                X_frame[j, :] = 0.0

    elif method == 'nearest':
        # For nearest, pick sample index closest to frame center
        # compute frame centers (seconds) and corresponding nearest sample indices
        frame_centers = (frame_edges[:-1] + frame_edges[1:]) / 2.0
        nearest_sample_idx = np.round(frame_centers * fs).astype(int)
        # clip to valid sample range
        nearest_sample_idx = np.clip(nearest_sample_idx, 0, N - 1)
        X_frame = X_mat[nearest_sample_idx, :]

        # return early with computed centers
        return X_frame, frame_centers, n_frames

    else:
        raise ValueError("method must be one of {'mean','sum','nearest'}")

    # compute frame centers for return
    frame_centers = (frame_edges[:-1] + frame_edges[1:]) / 2.0

    return X_frame, frame_centers, n_frames

def multi_kernel_deconv(emg, fs, force_mode, save_dir=None, max_iter=10, tol=1e-6):
    """
    Multi-kernel deconvolution entry point.

    Parameters
    ----------
    emg : array_like
        1D or 2D array of EMG data (N,) or (N, C)
    fs : float
        Sampling rate in Hz
    force_mode : str or list
        'single' or 'multi' per channel (or a list matching channel count)
    desktop_dir : str, optional
        Directory to save diagnostic figures
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance for IRLS (relative change)
    """
    # normalize emg shape to (N, C)
    emg = np.asarray(emg)
    if emg.ndim == 1:
        emg = emg[:, None]
    N, C = emg.shape

    # force_mode per channel
    if isinstance(force_mode, str):
        force_mode_list = [force_mode] * C
    else:
        assert len(force_mode) == C, "force_mode list must match channel count"
        force_mode_list = list(force_mode)

    ch_list = ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    for ch in range(C):
        s = emg[:, ch].astype(float)
        mode = force_mode_list[ch]

        # 1) estimate sigmas and build kernels
        sigmas = estimate_sigmas_from_gamma(s, fs, mode)
        K = build_kernels(sigmas, fs, mode)   # shape (Klen, nK)
        nK = K.shape[1]
        K_rev = K[::-1, :]

        # 2) define apply_A and apply_AT using convolution (avoid explicit Toeplitz)
        def apply_A(x):
            # use Fortran order so x blocks align with apply_AT block indexing
            x_mat = x.reshape(N, nK, order='F')
            recon = np.zeros(N)
            for kk in range(nK):
                recon += convolve(x_mat[:, kk], K[:, kk], mode='same')
            return recon

        def apply_AT(y):
            # accumulate kernel-correlations into a single vector using block layout
            grad = np.zeros(N * nK)
            for kk in range(nK):
                conv_res = convolve(y, K_rev[:, kk], mode='same')
                grad[kk*N:(kk+1)*N] = conv_res
            return grad
        
        n = N * nK
        adjoint_results = adjoint_test(apply_A, apply_AT, n=n, N=N, n_trials=5, tol=1e-8)

        # 3) estimate lam_max for alpha
        lam_max = estimate_lam_max(apply_A, apply_AT, N * nK, n_iters=12)
        alpha = 0.01 * lam_max
        print("DEBUG alpha:", alpha, type(alpha))

        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}. Check kernel construction and convolution symmetry.")

        # 4) IRLS using LinearOperator + LSMR (weights on augmented residual)
        n = N * nK

        # numeric hyperparameters
        eps_obj = 1e-12

        # warm start (optional)
        X = np.maximum(apply_AT(s), 0.0)

        conv_history = []

        for it in range(max_iter):
            # augmented residual r_aug = [A x - s; sqrt(alpha) * x]
            Ax = apply_A(X)                         # length N
            data_res = Ax - s                       # length N
            reg_res  = np.sqrt(alpha) * X           # length n
            r_aug = np.concatenate([data_res, reg_res])  # length m = N + n

            # adaptive small constant based on residual scale
            eps_reg = 1e-6 * max(1.0, np.max(np.abs(r_aug))) + 1e-12

            # IRLS weights from augmented residual: W_i = 1 / sqrt(|r_i| + eps_reg)
            w_aug = 1.0 / np.sqrt(np.abs(r_aug) + eps_reg)

            # define matvec / rmatvec for A2 = W * B
            def matvec(x):
                top = apply_A(x)                  # length N
                bottom = np.sqrt(alpha) * x       # length n
                Bx = np.concatenate([top, bottom])  # length m
                return w_aug * Bx                # apply W elementwise to the augmented vector

            def rmatvec(y):
                Wy = w_aug * y                    # elementwise multiply (length m)
                y_data = Wy[:N]
                y_reg  = Wy[N:]
                top_res = apply_AT(y_data)         # length n
                bot_res = np.sqrt(alpha) * y_reg   # length n
                return top_res + bot_res           # length n

            m = N + n
            A2 = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)

            # RHS is W * c_aug, where c_aug = [s; 0]
            c_aug = np.concatenate([s, np.zeros(n)])
            b2 = w_aug * c_aug
            # The default is maxiter = min(m, n), which A2 = m*n
            sol = lsmr(A2, b2, atol=1e-6, btol=1e-6)[0]

            # enforce non-negativity as in the paper
            X_new = np.maximum(sol, 0.0)

            # diagnostics: convergence
            diff_norm = np.linalg.norm(X_new - X)
            X_norm = np.linalg.norm(X) + eps_obj
            conv = diff_norm / X_norm
            conv_history.append(conv)

            X = X_new
            if conv < tol:
                break

        # 5) Reconstruct MUAP and compute CWF
        X_mat = X.reshape(N, nK, order='F')
        recon_from_applyA = apply_A(X)
        residual = s - recon_from_applyA

        # per-component time-domain reconstructions (units: µV if input s is µV)
        recon_comp = np.zeros((N, nK), dtype=float)
        for kk in range(nK):
            recon_comp[:, kk] = convolve(X_mat[:, kk], K[:, kk], mode='same')

        total_recon_check = np.sum(recon_comp, axis=1)

        # RMS of each kernel (units: same as K; K here is NOT amplitude-calibrated)
        rms_k = np.sqrt(np.mean(K**2, axis=0))
        # RMS of residual (units: µV)
        rms_res = np.sqrt(np.mean(residual**2))

        # CWF: rms-weighted sum across components
        # Note: numerically CWF has the same units as recon (µV) because X and K scale to match s (µV).
        CWF = X_mat.dot(rms_k)
        # Per-component CWF (units: µV) and per-second version (µV/s)
        CWF_comp = X_mat * rms_k[np.newaxis, :]

        # 6) plotting (only save if save_dir provided)
        if save_dir is not None:
            # Ensure top save_dir exists, then create dedicated plots subfolder.
            os.makedirs(save_dir, exist_ok=True)
            plots_dir = os.path.join(save_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            dpi_save = 150
            ch_label = ch_list[ch] if ch < len(ch_list) else f'Ch{ch+1}'

            # ensure X_frame exists for plotting (safe fallback)
            try:
                X_frame, frame_centers, n_frames = resample_to_frames(X_mat, fs, frame_rate=30.0, method='sum')
            except Exception:
                X_frame = np.zeros((0, nK))
                frame_centers = np.array([])
                n_frames = 0

            # a) kernels and firing patterns -> save to plots_dir
            fig, axes = plt.subplots(nK, 2, figsize=(10, 4*nK))
            if nK == 1:
                axes = axes.reshape(1, 2)
            t = np.linspace(-0.025, 0.025, K.shape[0])
            t_fp = np.arange(N) / fs
            for kk in range(nK):
                axes[kk, 0].plot(t, K[:, kk], linewidth=1.2)
                axes[kk, 0].set_xlabel('Time (s)')
                axes[kk, 0].set_ylabel('Amplitude (µV)')
                axes[kk, 0].set_title(f'{ch_label} — Kernel {kk+1}')
                axes[kk, 1].plot(t_fp, X_mat[:, kk], linewidth=1.2)
                axes[kk, 1].set_xlabel('Time (s)')
                axes[kk, 1].set_ylabel('Coefficient (a.u. per sample)')
                axes[kk, 1].set_title(f'{ch_label} — Firing Pattern {kk+1}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            fig.suptitle(f'{ch_label}: Kernels and Firing Patterns', fontsize=14, y=0.995)
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_Kernels_FP.png'), dpi=dpi_save)
            plt.close(fig)

            # b) per-component reconstructions (µV) + total -> plots_dir
            fig, axes_recon = plt.subplots(nK+1, 1, figsize=(10, 3*(nK+1)))
            if nK == 1:
                axes_recon = axes_recon.reshape(nK+1,)
            t_s = np.arange(N) / fs
            for kk in range(nK):
                axes_recon[kk].plot(t_s, recon_comp[:, kk], linewidth=1.0)
                axes_recon[kk].set_xlabel('Time (s)')
                axes_recon[kk].set_ylabel('Amplitude (µV)')
                axes_recon[kk].set_title(f'{ch_label}: reconstructed component {kk+1}')
            axes_recon[nK].plot(t_s, total_recon_check, 'r', linewidth=1.0, label='Total reconstruction (sum components)')
            axes_recon[nK].plot(t_s, s, 'k', linewidth=0.7, alpha=0.5, label='Original (ref)')
            axes_recon[nK].set_xlabel('Time (s)')
            axes_recon[nK].set_ylabel('Amplitude (µV)')
            axes_recon[nK].set_title(f'{ch_label}: Total reconstruction vs Original')
            axes_recon[nK].legend()
            axes_recon[nK].text(0.99, 0.95, f'RMS residual = {rms_res:.3f} µV',
                                transform=axes_recon[nK].transAxes, ha='right', va='top',
                                fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_Recon_Components.png'), dpi=dpi_save)
            plt.close(fig)

            # c) CWF (per-component and total) -> plots_dir
            fig = plt.figure(figsize=(10, 3*(nK+1)))
            for kk in range(nK):
                plt.subplot(nK+1, 1, kk+1)
                plt.plot(t_s, CWF_comp[:, kk], linewidth=1.0)
                plt.ylabel(f'Comp{kk+1} CWF (µV)')
                plt.title(f'{ch_label}: Component {kk+1} CWF')
            plt.subplot(nK+1, 1, nK+1)
            plt.plot(t_s, CWF, linewidth=1.2)
            plt.ylabel('Total CWF (µV)')
            plt.xlabel('Time (s)')
            plt.title(f'{ch_label}: Total CWF')
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_CWF_components.png'), dpi=dpi_save)
            plt.close(fig)

            # d) original vs recon and residual -> plots_dir
            fig, axes_or = plt.subplots(2, 1, figsize=(10, 6))
            axes_or[0].plot(t_s, s, 'k', linewidth=1.0, label='Original')
            axes_or[0].plot(t_s, recon_from_applyA, 'r', linewidth=1.0, label='Reconstruct')
            axes_or[0].set_xlabel('Time (s)')
            axes_or[0].set_ylabel('Amplitude (µV)')
            axes_or[0].set_title(f'{ch_label}: Original vs Reconstructed')
            axes_or[0].legend()
            axes_or[1].plot(t_s, residual)
            axes_or[1].set_xlabel('Time (s)')
            axes_or[1].set_ylabel('Amplitude (µV)')
            axes_or[1].set_title(f'{ch_label}: Residual')
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_Orig_Residual.png'), dpi=dpi_save)
            plt.close(fig)

            # e) convergence -> plots_dir
            fig = plt.figure(figsize=(8, 4))
            x = np.arange(len(conv_history)) + 1
            y = np.array(conv_history)
            plt.plot(x, y, '-o', markersize=4)
            plt.xlabel('IRLS iteration')
            plt.ylabel('||ΔX|| / ||X||')
            plt.title(f'{ch_label}: IRLS convergence')

            npts = len(y)
            max_labels = 50
            annot_every = 1 if npts <= max_labels else max(1, npts // max_labels)

            ymin, ymax = np.min(y), np.max(y)
            yrange = max(1e-12, ymax - ymin)

            for i, v in enumerate(y):
                if i % annot_every != 0:
                    continue
                xi = x[i]; yi = v
                offset_pts = 4 + (0 if yrange == 0 else int(10 * (v - ymin) / (yrange + 1e-12)))
                plt.annotate(f"{v:.3e}", xy=(xi, yi), xytext=(0, offset_pts), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8, rotation=25,
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.4))

            pad_up = 0.12 * yrange if yrange > 0 else 0.01
            pad_down = 0.06 * yrange if yrange > 0 else 0.0
            plt.ylim(ymin - pad_down, ymax + pad_up)

            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_Convergence.png'), dpi=dpi_save)
            plt.close(fig)

            # f) PSD comparison -> plots_dir
            fig = plt.figure(figsize=(10, 4))
            f1, P1 = welch(s, fs=fs)
            f2p, P2 = welch(recon_from_applyA, fs=fs)
            eps_db = 1e-20
            P1_db = 10.0 * np.log10(P1 + eps_db)
            P2_db = 10.0 * np.log10(P2 + eps_db)
            plt.plot(f1, P1_db, label='Original')
            plt.plot(f2p, P2_db, label='Reconstruct')
            plt.xlim([0, fs/2])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dB re µV^2/Hz)')
            plt.title(f'{ch_label}: PSD Original vs Recon (dB)')
            plt.legend()
            plt.grid(True, which='both', ls='--', lw=0.5)
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_PSD_dB.png'), dpi=dpi_save)
            plt.close(fig)

            # g) firing patterns, if frames available -> plots_dir
            if X_frame.size != 0:
                n_plots = X_frame.shape[1]
                fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
                axes = np.atleast_1d(axes)
                frame_x = np.arange(n_frames)
                for kk in range(n_plots):
                    axes[kk].plot(frame_x, X_frame[:, kk], linewidth=1.2)
                    axes[kk].set_xlabel('Bin (1 Bin = 1/30 s)')
                    axes[kk].set_ylabel('Coefficient (per Bin)')
                    axes[kk].set_title(f'{ch_label} — Firing Pattern (comp {kk+1})')
                    axes[kk].grid(True)
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                fig.suptitle(f'{ch_label}: Firing Patterns (Bin units)', fontsize=14, y=0.995)
                fig.savefig(os.path.join(plots_dir, f'{ch_label}_FiringPatterns_frames.png'), dpi=dpi_save)
                plt.close(fig)

        # Append results for this channel
        results.append({
            'sigmas': sigmas,
            'mode': mode,
            'K': K,
            'X': X_mat,
            'X_frame': X_frame,
            'recon_total': recon_from_applyA,
            'recon_components': recon_comp,
            'CWF': CWF,
            'CWF_comp': CWF_comp,
            'conv_history': conv_history,
            'alpha': alpha,
            'iterations': it+1,
            'residual': residual
        })

    # ------------- save once after all channels processed ---------------
    if save_dir is not None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        channel_names_to_use = ch_list[:C] if len(ch_list) >= C else [f'ch{i}' for i in range(C)]
        meta = {
            'fs': fs,
            'n_samples': N,
            'n_channels_input': C,
            'channels_expected': ch_list,
            'timestamp': datetime.now().isoformat()
        }
        # include mode in filename for clarity if available
        npz_name = os.path.join(save_dir, f'multi_kernel_deconv_results_{mode}_{ts}.npz')
        try:
            save_results_list_to_npz(results, npz_name, meta=meta, channel_names=channel_names_to_use, compress=True)
            print(f"Saved multi-kernel results to {npz_name}")
        except Exception as e_npz:
            print("Failed to save results npz:", e_npz)

    return results