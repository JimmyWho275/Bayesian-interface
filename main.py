"""
Doppler Shift Estimation via Particle Smoother
===============================================
Estimates time-varying radial velocity B_t from:
  - source  : known source signal (numpy array, float32/64)
  - received: sensor recording    (numpy array, float32/64)
  - sample_rate: sample rate in Hz (same for both signals)

Pipeline
--------
1. DopplerScorer.precompute()
     → For each candidate velocity v_k in [-10, +10] m/s (step 0.1),
       resample the source by alpha_k = c/(c+v_k),
       slide a window over the received signal,
       compute normalised cross-correlation peak.
     → Produces score_grid of shape (T_windows, N_velocities).

2. particle_filter()   [forward pass]
     → likelihood_fn interpolates score_grid at continuous particle velocities.
     → transition_fn propagates velocities via random walk.

3. particle_smoother() [backward pass, FFBS]
     → Produces smoothed trajectories p(B_t | A_{1:T}).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate, resample
from scipy.interpolate import interp1d
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

C_SOUND = 343.0   # m/s


# ─────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    particles:    np.ndarray   # (T, N)
    weights:      np.ndarray   # (T, N)
    ess:          np.ndarray   # (T,)
    log_evidence: float


@dataclass
class SmootherResult:
    trajectories:  np.ndarray  # (M, T)
    mean:          np.ndarray  # (T,)
    std:           np.ndarray  # (T,)
    q05:           np.ndarray  # (T,)
    q95:           np.ndarray  # (T,)
    filter_result: FilterResult


# ─────────────────────────────────────────────────────────────────
#  Doppler Scorer
# ─────────────────────────────────────────────────────────────────

class DopplerScorer:
    """
    Pre-computes a score grid:  score_grid[t, k] = normalised xcorr peak
    between the t-th received window and source resampled by alpha(v_k).

    Parameters
    ----------
    source       : source signal array
    sample_rate  : Hz
    c            : speed of sound (m/s)
    v_min, v_max : velocity search range (m/s)
    v_res        : velocity resolution (m/s)
    gamma        : likelihood sharpness — higher = more peaked distribution
    """

    def __init__(
        self,
        source:      np.ndarray,
        sample_rate: float,
        c:           float = C_SOUND,
        v_min:       float = -10.0,
        v_max:       float =  10.0,
        v_res:       float =  0.1,
        gamma:       float =  15.0,
    ):
        self.source      = source.astype(np.float64)
        self.sr          = sample_rate
        self.c           = c
        self.v_grid      = np.arange(v_min, v_max + v_res / 2, v_res)
        self.gamma       = gamma
        self.score_grid  = None   # set after precompute()
        self.n_windows   = None
        self.window_samp = None
        self.hop_samp    = None

    # ── helpers ───────────────────────────────────────────────────

    def alpha(self, v: float) -> float:
        """Doppler resampling factor: received is compressed/stretched by alpha."""
        return self.c / (self.c + v)

    def _resampled_source(self, v: float) -> np.ndarray:
        """Return source resampled to simulate reception at radial velocity v."""
        a        = self.alpha(v)
        new_len  = max(1, int(round(len(self.source) * a)))
        return resample(self.source, new_len)

    @staticmethod
    def _norm_xcorr_peak(window: np.ndarray, template: np.ndarray) -> float:
        """
        Normalised cross-correlation peak in [0, 1].
        max |xcorr(window, template)| / (||window|| * ||template||)
        """
        norm = np.linalg.norm(window) * np.linalg.norm(template)
        if norm < 1e-12:
            return 0.0
        xc = correlate(window, template, mode='full')
        return float(np.max(np.abs(xc)) / norm)

    # ── main pre-computation ──────────────────────────────────────

    def precompute(
        self,
        received:       np.ndarray,
        window_sec:     float,
        hop_sec:        float,
    ) -> np.ndarray:
        """
        Build score_grid[t, k] for all windows t and velocities v_k.

        Parameters
        ----------
        received    : received sensor signal
        window_sec  : window duration in seconds
        hop_sec     : hop (step) between windows in seconds

        Returns
        -------
        score_grid : np.ndarray, shape (T_windows, N_velocities), values in [0, 1]
        """
        received        = received.astype(np.float64)
        window_samp     = int(round(window_sec * self.sr))
        hop_samp        = int(round(hop_sec    * self.sr))
        n_windows       = max(1, (len(received) - window_samp) // hop_samp + 1)

        self.window_samp = window_samp
        self.hop_samp    = hop_samp
        self.n_windows   = n_windows

        V           = len(self.v_grid)
        score_grid  = np.zeros((n_windows, V))

        print(f"Pre-computing score grid: {n_windows} windows × {V} velocities ...")

        # Pre-resample source for each velocity candidate once
        templates = [self._resampled_source(v) for v in self.v_grid]

        for k, template in enumerate(templates):
            if k % 20 == 0:
                print(f"  velocity {self.v_grid[k]:+.1f} m/s  ({k}/{V})")
            for t in range(n_windows):
                start             = t * hop_samp
                window            = received[start : start + window_samp]
                score_grid[t, k]  = self._norm_xcorr_peak(window, template)

        self.score_grid = score_grid
        print("Pre-computation done.\n")
        return score_grid

    # ── likelihood for particle filter ────────────────────────────

    def likelihood(self, t: int, particles: np.ndarray) -> np.ndarray:
        """
        For time frame t, return likelihood p(A_t | v^(i)) for all particles.

        Interpolates score_grid[t, :] at each particle's continuous velocity,
        then converts via exponential: likelihood = exp(gamma * score).

        Parameters
        ----------
        t         : time frame index
        particles : (N,) array of particle velocities in m/s

        Returns
        -------
        likelihoods : (N,) array, unnormalised
        """
        scores_at_t  = self.score_grid[t]                   # (V,)
        interp_fn    = interp1d(
            self.v_grid, scores_at_t,
            kind='linear', bounds_error=False, fill_value=0.0
        )
        scores       = np.clip(interp_fn(particles), 0.0, 1.0)
        return np.exp(self.gamma * scores)


# ─────────────────────────────────────────────────────────────────
#  Particle Filter + Smoother
# ─────────────────────────────────────────────────────────────────

def _systematic_resample(weights: np.ndarray) -> np.ndarray:
    N       = len(weights)
    pos     = (np.arange(N) + np.random.uniform()) / N
    cumsum  = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, pos)


def particle_filter(
    scorer:             DopplerScorer,
    n_particles:        int   = 1000,
    transition_std:     float = 0.2,     # m/s — how fast velocity can change per step
    v_min:              float = -10.0,
    v_max:              float =  10.0,
    resample_threshold: float = 0.5,
) -> FilterResult:
    """
    Bootstrap Particle Filter over the pre-computed score grid.

    Parameters
    ----------
    scorer           : DopplerScorer with precompute() already called
    n_particles      : number of particles N
    transition_std   : std of random-walk transition on velocity (m/s per step)
    v_min, v_max     : hard velocity bounds (particles clipped to this range)
    resample_threshold: resample when ESS < threshold * N

    Returns
    -------
    FilterResult
    """
    T    = scorer.n_windows
    N    = n_particles
    rng  = np.random.default_rng()

    all_particles = np.zeros((T, N))
    all_weights   = np.zeros((T, N))
    ess_trace     = np.zeros(T)
    log_evidence  = 0.0

    # Prior: uniform over velocity range
    particles = rng.uniform(v_min, v_max, N)

    for t in range(T):

        # 1. Propagate (random walk on velocity)
        if t > 0:
            particles = particles + rng.normal(0, transition_std, N)
            particles = np.clip(particles, v_min, v_max)

        # 2. Weight by likelihood from scorer
        raw_w = scorer.likelihood(t, particles)
        raw_w = np.maximum(raw_w, 1e-300)

        log_evidence += np.log(raw_w.sum() / N)

        # 3. Normalise
        weights = raw_w / raw_w.sum()

        # 4. ESS
        ess = 1.0 / np.sum(weights ** 2)
        ess_trace[t] = ess

        # 5. Resample if needed
        if ess < resample_threshold * N:
            idx       = _systematic_resample(weights)
            particles = particles[idx]
            weights   = np.ones(N) / N

        all_particles[t] = particles
        all_weights[t]   = weights

    return FilterResult(
        particles=all_particles,
        weights=all_weights,
        ess=ess_trace,
        log_evidence=log_evidence,
    )


def particle_smoother(
    filter_result:  FilterResult,
    transition_std: float = 0.2,
    n_trajectories: int   = 300,
) -> SmootherResult:
    """
    Forward Filter Backward Sampler (FFBS).

    Walks backwards through the filtered particles, re-weighting at each step
    by the transition density p(B_{t+1}^* | B_t^(i)).

    Parameters
    ----------
    filter_result  : output of particle_filter()
    transition_std : must match the std used in particle_filter()
    n_trajectories : number of smoothed trajectory samples M

    Returns
    -------
    SmootherResult
    """
    particles = filter_result.particles   # (T, N)
    weights   = filter_result.weights     # (T, N)
    T, N      = particles.shape

    trajectories = np.zeros((n_trajectories, T))

    for m in range(n_trajectories):
        traj = np.zeros(T)

        # Sample final state from filtered distribution
        idx      = np.random.choice(N, p=weights[T - 1])
        traj[T-1] = particles[T - 1, idx]

        # Backward pass
        for t in range(T - 2, -1, -1):
            next_v = traj[t + 1]

            # Transition density: p(next_v | B_t^(i))
            diff        = (next_v - particles[t]) / transition_std
            trans_probs = np.exp(-0.5 * diff ** 2)

            back_w = weights[t] * trans_probs
            total  = back_w.sum()

            if total < 1e-300:
                back_w = weights[t]   # fallback
            else:
                back_w /= total

            idx     = np.random.choice(N, p=back_w)
            traj[t] = particles[t, idx]

        trajectories[m] = traj

    mean = trajectories.mean(axis=0)
    std  = trajectories.std(axis=0)
    q05  = np.percentile(trajectories,  5, axis=0)
    q95  = np.percentile(trajectories, 95, axis=0)

    return SmootherResult(
        trajectories=trajectories,
        mean=mean,
        std=std,
        q05=q05,
        q95=q95,
        filter_result=filter_result,
    )


# ─────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────

def plot_spectrograms(
    source:      np.ndarray,
    received:    np.ndarray,
    sample_rate: float,
    nperseg:     int   = 256,
    noverlap:    int   = 224,
    f_max:       float | None = None,
    save_path:   str   | None = "doppler_spectrograms.png",
):
    """
    Plot spectrograms of source and received signals side by side.

    Parameters
    ----------
    source, received : signal arrays
    sample_rate      : Hz
    nperseg          : STFT window length (samples) — controls time/freq resolution
    noverlap         : overlap between windows (samples)
    f_max            : upper frequency limit for display (Hz); None = sr/2
    save_path        : where to save the figure
    """
    from scipy.signal import spectrogram

    def _compute_spec(sig):
        f, t, Sxx = spectrogram(sig, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        Sxx_db    = 10 * np.log10(Sxx + 1e-12)   # to dB, avoid log(0)
        return f, t, Sxx_db

    f_src, t_src, S_src = _compute_spec(source)
    f_rcv, t_rcv, S_rcv = _compute_spec(received)

    # Frequency display limit
    f_lim    = f_max if f_max is not None else sample_rate / 2
    f_mask_s = f_src <= f_lim
    f_mask_r = f_rcv <= f_lim

    # Shared colour scale across both panels
    vmin = min(S_src[f_mask_s].min(), S_rcv[f_mask_r].min())
    vmax = max(S_src[f_mask_s].max(), S_rcv[f_mask_r].max())

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle('Signal Spectrograms', fontsize=13, fontweight='bold')

    for ax, (f, t, S, f_mask, title) in zip(axes, [
        (f_src, t_src, S_src, f_mask_s, 'Source signal'),
        (f_rcv, t_rcv, S_rcv, f_mask_r, 'Received signal (sensor)'),
    ]):
        im = ax.pcolormesh(
            t, f[f_mask], S[f_mask],
            shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label='Power (dB)')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, f_lim)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectrograms saved → {save_path}")
    plt.show()


def plot_results(
    smoother_result: SmootherResult,
    scorer:          DopplerScorer,
    true_velocity:   np.ndarray | None = None,
    time_axis:       np.ndarray | None = None,
    n_traj_plot:     int = 40,
    save_path:       str | None = "doppler_results.png",
):
    fr   = smoother_result.filter_result
    T    = fr.particles.shape[0]
    t    = time_axis if time_axis is not None else np.arange(T)

    fig  = plt.figure(figsize=(14, 12))
    gs   = gridspec.GridSpec(4, 1, hspace=0.5)

    # ── Panel 1: Score grid heatmap ───────────────────────────────
    ax1  = fig.add_subplot(gs[0])
    ext  = [t[0], t[-1], scorer.v_grid[0], scorer.v_grid[-1]]
    im   = ax1.imshow(
        scorer.score_grid.T, aspect='auto', origin='lower',
        extent=ext, cmap='inferno', vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax1, label='Normalised xcorr score')
    ax1.set_title('Score Grid  —  $p(A_t \mid v_k)$ [normalised xcorr]', fontsize=11)
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_xlabel('Time (s)')

    # ── Panel 2: Smoothed posterior ───────────────────────────────
    ax2  = fig.add_subplot(gs[1])
    for i in range(min(n_traj_plot, smoother_result.trajectories.shape[0])):
        ax2.plot(t, smoother_result.trajectories[i],
                 color='steelblue', alpha=0.07, linewidth=0.8)

    ax2.fill_between(t, smoother_result.q05, smoother_result.q95,
                     color='steelblue', alpha=0.25, label='90% credible interval')
    ax2.plot(t, smoother_result.mean,
             color='steelblue', linewidth=2.0, label='Smoothed mean')

    if true_velocity is not None:
        ax2.plot(t, true_velocity, 'k--', linewidth=1.5, label='True velocity')

    ax2.set_title('Smoothed Posterior  $p(v_t \mid A_{1:T})$', fontsize=11)
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Filter vs Smoother ───────────────────────────────
    ax3  = fig.add_subplot(gs[2])
    filtered_mean = (fr.particles * fr.weights).sum(axis=1)
    ax3.plot(t, filtered_mean,
             color='tomato', linewidth=1.8, linestyle='--', label='Filtered mean')
    ax3.plot(t, smoother_result.mean,
             color='steelblue', linewidth=1.8, label='Smoothed mean')

    if true_velocity is not None:
        ax3.plot(t, true_velocity, 'k--', linewidth=1.5, label='True velocity')

    ax3.set_title('Filter vs Smoother', fontsize=11)
    ax3.set_ylabel('Velocity (m/s)')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: ESS trace ────────────────────────────────────────
    ax4  = fig.add_subplot(gs[3])
    N    = fr.particles.shape[1]
    ax4.plot(t, fr.ess, color='seagreen', linewidth=1.5)
    ax4.axhline(N * 0.5, color='tomato', linestyle='--',
                linewidth=1.0, label=f'Resample threshold ({N//2})')
    ax4.set_title('Effective Sample Size (ESS)', fontsize=11)
    ax4.set_ylabel('ESS')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylim(0, N * 1.05)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────
#  Synthetic demo
# ─────────────────────────────────────────────────────────────────

def make_synthetic_signals(
    sample_rate:    float = 8000.0,
    duration_src:   float = 0.5,    # source clip duration (s)
    duration_recv:  float = 10.0,   # received signal duration (s)
    freq:           float = 440.0,  # source tone frequency (Hz)
    noise_std:      float = 0.05,
    rng:            np.random.Generator | None = None,
):
    """
    Synthetic test: source is a tone, received is a Doppler-shifted+noisy version.
    True velocity follows a smooth curve (sinusoidal).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sr   = sample_rate
    t_s  = np.linspace(0, duration_src,  int(sr * duration_src),  endpoint=False)
    t_r  = np.linspace(0, duration_recv, int(sr * duration_recv), endpoint=False)

    # Source: simple tone with envelope
    source   = np.sin(2 * np.pi * freq * t_s) * np.hanning(len(t_s))

    # True velocity: slow sinusoidal variation ±5 m/s
    true_vel_fn = lambda t: 5.0 * np.sin(2 * np.pi * t / duration_recv)
    true_v_cont = true_vel_fn(t_r)

    # Build received signal by stitching Doppler-shifted source chunks
    received = np.zeros(len(t_r))
    chunk    = int(sr * duration_src * 0.5)   # half-source chunks
    pos      = 0
    while pos < len(t_r) - chunk:
        v_local = true_v_cont[pos + chunk // 2]
        alpha   = C_SOUND / (C_SOUND + v_local)
        new_len = max(1, int(round(len(source) * alpha)))
        s_shifted = resample(source, new_len)
        end = min(pos + len(s_shifted), len(received))
        received[pos:end] += s_shifted[:end - pos]
        pos += chunk

    received += rng.normal(0, noise_std, len(received))
    received /= np.max(np.abs(received) + 1e-8)

    return source, received, sr, true_vel_fn


def run_demo():
    rng = np.random.default_rng(42)

    # ── Generate synthetic signals ────────────────────────────────
    print("Generating synthetic signals ...")
    source, received, sr, true_vel_fn = make_synthetic_signals(rng=rng)

    # ── Build scorer and pre-compute score grid ───────────────────
    scorer = DopplerScorer(
        source      = source,
        sample_rate = sr,
        c           = C_SOUND,
        v_min       = -10.0,
        v_max       =  10.0,
        v_res       =  0.1,
        gamma       =  15.0,   # sharpness of likelihood — tune this
    )

    scorer.precompute(
        received    = received,
        window_sec  = 0.3,   # window duration  — tune: longer = better v resolution
        hop_sec     = 0.1,   # hop between windows → time resolution of output
    )

    T         = scorer.n_windows
    hop_sec   = scorer.hop_samp / sr
    time_axis = np.arange(T) * hop_sec
    true_v    = true_vel_fn(time_axis)

    # ── Forward particle filter ───────────────────────────────────
    print("Running particle filter ...")
    filter_result = particle_filter(
        scorer          = scorer,
        n_particles     = 1000,
        transition_std  = 0.3,    # m/s per hop — tune to match velocity dynamics
        resample_threshold = 0.5,
    )
    print(f"  log evidence : {filter_result.log_evidence:.2f}")
    print(f"  mean ESS     : {filter_result.ess.mean():.1f} / 1000")

    # ── Backward smoother ─────────────────────────────────────────
    print("Running backward smoother (FFBS) ...")
    smoother_result = particle_smoother(
        filter_result   = filter_result,
        transition_std  = 0.3,
        n_trajectories  = 300,
    )

    # ── Report ────────────────────────────────────────────────────
    filt_mean    = (filter_result.particles * filter_result.weights).sum(axis=1)
    filter_rmse  = np.sqrt(np.mean((filt_mean - true_v) ** 2))
    smooth_rmse  = np.sqrt(np.mean((smoother_result.mean - true_v) ** 2))
    print(f"\n  Filter RMSE  : {filter_rmse:.4f} m/s")
    print(f"  Smoother RMSE: {smooth_rmse:.4f} m/s")

    # ── Plot spectrograms ─────────────────────────────────────────
    plot_spectrograms(
        source      = source,
        received    = received,
        sample_rate = sr,
        save_path   = "/mnt/user-data/outputs/doppler_spectrograms.png",
    )

    # ── Plot inference results ────────────────────────────────────
    plot_results(
        smoother_result,
        scorer,
        true_velocity = true_v,
        time_axis     = time_axis,
        save_path     = "/mnt/user-data/outputs/doppler_results.png",
    )

    return scorer, filter_result, smoother_result


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer, filter_result, smoother_result = run_demo()