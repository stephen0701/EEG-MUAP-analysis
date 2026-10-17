# %%
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
from signal import signal
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import welch, convolve
from scipy.sparse.linalg import lsmr, LinearOperator

def estimate_sigmas_from_gamma(signal, fs, mode):
    """
    Estimate Gaussian kernel sigmas from EMG PSD.
    mode: 'multi' -> linear fit on gamma vs f^2 -> symmetric two kernels
          'single' -> quadratic fit, return three percentile-based sigmas
    """
    nperseg = min(4096, len(signal))
    noverlap = nperseg // 2
    nfft = max(nperseg, 2**int(np.ceil(np.log2(nperseg))))

    f, pxx = welch(signal, fs=fs, window='hamming',
                   nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=0)

    mask = f > 0
    f = f[mask]; pxx = pxx[mask]
    if f.size == 0:
        warnings.warn("No positive frequency bins from welch; returning default sigmas.")
        return [0.003, 0.003] if mode == 'multi' else [0.003, 0.003, 0.003]

    #f_med = np.median(f); f_std = np.std(f)
    #mask2 = (f >= (f_med - f_std)) & (f <= (f_med + 2*f_std))
    #f_sel = f[mask2]; pxx_sel = pxx[mask2]
    f_sel = f; pxx_sel = pxx

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
        print(f"[DEBUG] multi mode: slope={slope:.6e}")
        if slope >= 0:
            warnings.warn("slope >= 0 from gamma vs f^2 fit — check PSD or selection band.")
        sigma2 = -slope / (4 * np.pi**2)
        sigma = np.sqrt(max(sigma2, eps))
        print("DEBUG sigmas:", sigma)
        return np.array([sigma, sigma]), {'f_sel': f_sel, 'pxx_sel': pxx_sel, 'f0_list': [], 'pctls': []}
    else:
        p = np.polyfit(f2, gamma, 2)  # a f^4 + b f^2 + c
        a, b = p[0], p[1]
        print(f"[DEBUG] single mode: a={a:.6e}, b={b:.6e}")
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

        f0_list = []
        slope_list = []
        fallback_list = []

        for pc in pctls:
            # interpolate percentile to get a smooth f0 (avoid discrete index jumps)
            f0 = np.interp(pc, cdf, f_sel)
            f0_list.append(f0)

            # compute derivative of polyfit (gamma = a*(f^2)^2 + b*(f^2) + c)
            # derivative wrt (f^2) is: dgamma/d(f^2) = 2*a*(f^2) + b
            f0_sq = f0**2
            slope = 2.0 * a * f0_sq + b
            slope_list.append(slope)

            fallback = False
            if slope >= 0:
                warnings.warn(f"slope >= 0 from gamma fit at PSD-weighted percentile {pc:.2f} — using fallback linear fit.")
                fallback = True
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
            fallback_list.append(fallback)

        print(f"[DEBUG] f0_list={f0_list}")
        print(f"[DEBUG] slope_list={slope_list}")
        print(f"[DEBUG] fallback_list={fallback_list}")
        print(f"[DEBUG] sigmas:", sigmas)

        return np.array(sigmas), {'f_sel': f_sel, 'pxx_sel': pxx_sel, 'f0_list': f0_list, 'pctls': pctls}

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

def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path

class AlternativeProcessor:
    """
    Simple preprocessing pipeline:
    1. Detrend (remove linear trend)
    2. Apply notch filter (remove powerline noise)
    3. Apply highpass filter (remove low-frequency drift)
    4. Apply lowpass filter (remove high-frequency noise)
    """
    def __init__(self, fs=1000, notch_f0=60, hp_fc=40, lp_fc=450):
        from scipy.signal import iirnotch, butter
        self.fs = fs
        # Design notch filter
        self.b_notch, self.a_notch = iirnotch(notch_f0, 30, fs)
        # Design highpass filter
        nyq = 0.5 * fs
        hp_wn = hp_fc / nyq
        self.b_hp, self.a_hp = butter(4, hp_wn, btype='highpass', output='ba')
        # Design lowpass filter
        lp_wn = lp_fc / nyq
        self.b_lp, self.a_lp = butter(4, lp_wn, btype='lowpass', output='ba')

    def process(self, x):
        """Apply detrend, notch, highpass, lowpass filters."""
        from scipy.signal import detrend, filtfilt
        y = detrend(x)
        y = filtfilt(self.b_notch, self.a_notch, y)
        y = filtfilt(self.b_hp, self.a_hp, y)
        y = filtfilt(self.b_lp, self.a_lp, y)
        return y

def multi_kernel_deconv(emg, fs, force_mode, save_dir=None, max_iter=10, tol=1e-6, ch_labels=None):
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
    save_dir : str, optional
        Directory to save diagnostic figures
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance for IRLS (relative change)
    ch_labels : list of str, optional
        Channel labels for plotting
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

    ch_list = ch_labels if ch_labels else ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    for ch in range(C):
        s = emg[:, ch].astype(float)
        mode = force_mode_list[ch]
        ch_label = ch_list[ch] if ch < len(ch_list) else f'Ch{ch+1}'

        # 1) estimate sigmas and build kernels
        sigmas, psd_info = estimate_sigmas_from_gamma(s, fs, mode)
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
            
            # Plot percentile lines and annotate frequency values (only for 'single' mode)
            if mode == 'single' and psd_info and psd_info.get('f0_list'):
                f0_list = psd_info['f0_list']
                pctls = psd_info['pctls']
                for i, (pc, f0) in enumerate(zip(pctls, f0_list)):
                    plt.axvline(f0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                               label=f'Percentile {pc:.2f}: {f0:.1f} Hz')
            
            plt.xlim([0, fs/2])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (dB re µV^2/Hz)')
            plt.title(f'{ch_label}: PSD Original vs Recon (dB)')
            plt.legend()
            plt.grid(True, which='both', ls='--', lw=0.5)
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, f'{ch_label}_PSD_dB.png'), dpi=dpi_save)
            plt.close(fig)

        # Append results for this channel
        results.append({
            'sigmas': sigmas,
            'mode': mode,
            'K': K,
            'X': X_mat,
            'recon_total': recon_from_applyA,
            'recon_components': recon_comp,
            'CWF': CWF,
            'CWF_comp': CWF_comp,
            'conv_history': conv_history,
            'alpha': alpha,
            'iterations': it+1,
            'residual': residual
        })

    return results

# ============================================================================
# High-level processing functions
# ============================================================================

def process_single_dataset(
    dataset_path,
    muscles=None,
    fs=1000,
    modes=('single', 'multi'),
    save_results=True,
    verbose=True
):
    """
    Preprocess EMG and run multi-kernel deconvolution for a single dataset folder.
    Reads aligned EMG CSV files (*_align.csv) generated by movement_state_detector.
    
    Parameters
    ----------
    dataset_path : str
        Path to dataset directory containing EMG subfolder
    muscles : list of str, optional
        List of muscle channel names to process
    fs : float
        Sampling frequency in Hz
    modes : tuple of str
        Deconvolution modes to run ('single', 'multi')
    save_results : bool
        Whether to save plots and NPZ files
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Results per mode with pickle and npz paths
    """
    import glob
    import pickle
    import pandas as pd
    
    if muscles is None:
        muscles = ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF']
    
    preprocessor = AlternativeProcessor(fs=fs)
    
    # Read aligned EMG CSV file directly (output from movement_state_detector)
    emg_folder = os.path.join(dataset_path, 'EMG')
    align_files = glob.glob(os.path.join(emg_folder, '*_align.csv'))
    
    if not align_files:
        raise FileNotFoundError(f"No *_align.csv file found in {emg_folder}. Please run movement_state_detector first.")
    
    # Use the first align file found
    align_file = align_files[0]
    if verbose:
        print(f"Reading aligned EMG file: {os.path.basename(align_file)}")
    
    df0 = pd.read_csv(align_file)
    
    # Map LGL→LGA and RGL→RGA
    channel_mapping = {'LGL': 'LGA', 'RGL': 'RGA'}
    df0 = df0.rename(columns=channel_mapping)
    
    available = [m for m in muscles if m in df0.columns]
    if not available:
        raise ValueError(f"No expected channels found in {align_file}; available: {df0.columns.tolist()}")
    
    # Preprocess each channel
    clean_signals = []
    for ch in available:
        arr = df0[ch].astype(float).values
        clean_signals.append(preprocessor.process(arr))
    clean_signals = np.column_stack(clean_signals)

    # NaN protection
    clean_df = pd.DataFrame(clean_signals, columns=available)
    if clean_df.isna().values.any():
        clean_df = clean_df.interpolate(axis=0).bfill().ffill().fillna(0.0)
    clean_signals = clean_df.to_numpy()

    results = {}
    top_save = os.path.join(dataset_path, 'multi-kernel-deconv')
    ensure_dir(top_save)
    
    for mode in modes:
        mode_dir = os.path.join(top_save, mode)
        ensure_dir(mode_dir)
        
        if verbose:
            print(f"Running multi-kernel deconvolution ({mode}) ...")
        
        deconv_result = multi_kernel_deconv(
            clean_signals,
            fs,
            force_mode=mode,
            save_dir=mode_dir if save_results else None,
            max_iter=10,
            tol=1e-6,
            ch_labels=available
        )
        
        # Save pickle summary
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pkl_path = os.path.join(mode_dir, f'results_{mode}_{ts}.pkl')
        with open(pkl_path, 'wb') as pf:
            pickle.dump(deconv_result, pf, protocol=pickle.HIGHEST_PROTOCOL)
        
        results[mode] = {'pickle': pkl_path, 'n_channels': len(deconv_result)}
    
    return results

def process_all_datasets(
    parent_dir,
    muscles=None,
    fs=1000,
    modes=('single', 'multi'),
    save_results=True,
    verbose=True
):
    """
    Batch process all path2_* datasets in parent_dir.
    
    Parameters
    ----------
    parent_dir : str
        Parent directory containing path2_* subdirectories
    muscles : list of str, optional
        List of muscle channel names to process
    fs : float
        Sampling frequency in Hz
    modes : tuple of str
        Deconvolution modes to run ('single', 'multi')
    save_results : bool
        Whether to save plots and NPZ files
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Master summary with results for all datasets
    """
    import glob
    
    pattern = os.path.join(parent_dir, 'path2*')
    data_dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    
    if not data_dirs:
        print(f"No path2* directories found under {parent_dir}")
        return {}

    master_results = {
        'meta': {
            'parent_dir': parent_dir,
            'fs': fs,
            'muscles': muscles if muscles else ['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF'],
            'modes': modes,
            'timestamp': datetime.now().isoformat()
        },
        'datasets': {}
    }

    for data_dir in data_dirs:
        print(f"\n=== Processing dataset: {data_dir} ===")
        try:
            ds_results = process_single_dataset(
                data_dir,
                muscles=muscles,
                fs=fs,
                modes=modes,
                save_results=save_results,
                verbose=verbose
            )
            master_results['datasets'][os.path.basename(data_dir)] = {
                'success': True,
                'files': ds_results
            }
        except Exception as e:
            print(f"Error processing {data_dir}: {e}")
            master_results['datasets'][os.path.basename(data_dir)] = {
                'success': False,
                'error': str(e)
            }

    # Save master summary
    master_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_fn = os.path.join(parent_dir, f"master_results_summary_{master_ts}.json")
    with open(master_fn, 'w', encoding='utf-8') as mf:
        json.dump(master_results, mf, ensure_ascii=False, indent=2)
    print(f"All done. Master summary saved to {master_fn}")
    return master_results

def main():
    """
    Main entry point for EMG multi-kernel deconvolution processing.
    
    Usage examples:
    
    1. Process single dataset:
       python EMG_to_MUAP.py
       (set the RESEARCH_DATASET environment variable)
       
    2. Process all datasets in parent directory:
       Uncomment process_all_datasets() call
    """
    # Process the explicitly configured, authorized trial folder.
    DATASET_PATH = os.environ["RESEARCH_DATASET"]
    
    print(f"Processing single dataset: {DATASET_PATH}")
    results = process_single_dataset(
        dataset_path=DATASET_PATH,
        muscles=['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF'],
        fs=1000,
        modes=('single', 'multi'),
        save_results=True,
        verbose=True
    )
    
    print("\n=== Processing Complete ===")
    print(f"Results saved to: {DATASET_PATH}/multi-kernel-deconv/")
    for mode, info in results.items():
        print(f"  {mode} mode:")
        print(f"    - Pickle: {info['pickle']}")
        print(f"    - Channels: {info['n_channels']}")
    
    # To process all datasets, uncomment below:
    # PARENT_DIR = os.environ["RESEARCH_DATA_ROOT"]
    # master_results = process_all_datasets(
    #     parent_dir=PARENT_DIR,
    #     muscles=['LGA','RGA','LTA','RTA','LBF','RBF','LRF','RRF'],
    #     fs=1000,
    #     modes=('single', 'multi'),
    #     save_results=True,
    #     verbose=True
    # )

if __name__ == "__main__":
    main()

# %%
