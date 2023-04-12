from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from .peq import ParametricEqualizer
from .praat import PraatAugment

class Config():
    def __init__(self):
        self.sr = 16000

        #all STFT hyperparameters
        self.mel = 80
        self.mel_win = 1024
        self.mel_hop = 256
        
        # self.mel_win_fn = 'hann'
        # self.mel_fmin = 0
        # self.mel_fmax = 8000
        
        # augment
        self.cutoff_lowpass = 60
        self.cutoff_highpass = 10000
        self.q_min = 2
        self.q_max = 5
        self.num_peak = 8
        
        # self.num_code = 32
        self.formant_shift = 1.4
        self.pitch_shift = 2.
        self.pitch_range = 1.5
        self.g_min = -12
        self.g_max = 12
        # # pitch consistency
        # self.cqt_shift_min = -12
        # self.cqt_shift_max = 12
        # # linguistic informations
        # self.kappa = 0.1

        
class Augmentor():
    def __init__(self):
        self.config = Config()
        self.aug = Augment(self.config)
        
    def sample_like(self, signal: torch.Tensor) -> List[torch.Tensor]:
        """Sample augmentation parameters.
        Args:
            signal: [torch.float32; [B, T]], speech signal.
        Returns:
            augmentation parameters.
        """
        # [B]
        bsize, _ = signal.shape
        def sampler(ratio):
            shifts = torch.rand(bsize, device=signal.device) * (ratio - 1.) + 1.
            # flip
            flip = torch.rand(bsize) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts
        # sample shifts
        fs = sampler(self.config.formant_shift)
        ps = sampler(self.config.pitch_shift)
        pr = sampler(self.config.pitch_range)
        # parametric equalizer
        peaks = self.config.num_peak
        # quality factor
        power = torch.rand(bsize, peaks + 2, device=signal.device)
        # gains
        g_min, g_max = self.config.g_min, self.config.g_max
        gain = torch.rand(bsize, peaks + 2, device=signal.device) * (g_max - g_min) + g_min
        return fs, ps, pr, power, gain

    @torch.no_grad()
    def augment(self, signal: torch.Tensor) -> torch.Tensor:
        """Augment the speech.
        Args:
            signal: [torch.float32; [B, T]], segmented speech.
        Returns:
            [torch.float32; [B, T]], speech signal.
        """
        # B
        bsize, _ = signal.shape
        saves = None
        while saves is None or len(saves) < bsize:
            # [B] x 4
            fshift, pshift, prange, power, gain = self.sample_like(signal)
            # [B, T]
            out = self.aug.forward(signal, pshift, prange, fshift, power, gain)
            # for covering unexpected NaN
            nan = out.isnan().any(dim=-1)
            if not nan.all():
                # save the outputs for not-nan inputs
                if saves is None:
                    saves = out[~nan]
                else:
                    saves = torch.cat([saves, out[~nan]], dim=0)
        # [B, T]
        return saves[:bsize]
        

class Augment(nn.Module):
    """Waveform augmentation.
    """

    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: configurations.
        """
        super().__init__()
        self.config = config
        self.praat = PraatAugment(config)  # lex modify to config
        self.peq = ParametricEqualizer(
            config.sr, config.mel_win)
        self.register_buffer(
            'window',
            torch.hann_window(config.mel_win),
            persistent=False)
        f_min, f_max, peaks = \
            config.cutoff_lowpass, \
            config.cutoff_highpass, config.num_peak
        # peaks except frequency min and max
        self.register_buffer(
            'peak_centers',
            f_min * (f_max / f_min) ** (torch.arange(peaks + 2)
                                        [1:-1] / (peaks + 1)),
            persistent=False)

    def forward(self,
                wavs: torch.Tensor,
                pitch_shift: Optional[torch.Tensor] = None,
                pitch_range: Optional[torch.Tensor] = None,
                formant_shift: Optional[torch.Tensor] = None,
                quality_power: Optional[torch.Tensor] = None,
                gain: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Augment the audio signal, random pitch, formant shift and PEQ.
        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            pitch_shift: [torch.float32; [B]], pitch shifts.
            pitch_range: [torch.float32; [B]], pitch ranges.
            formant_shift: [torch.float32; [B]], formant shifts.
            quality_power: [torch.float32; [B, num_peak + 2]],
                exponents of quality factor, for PEQ.
            gain: [torch.float32; [B, num_peak + 2]], gain in decibel.
        Returns:
            [torch.float32; [B, T]], augmented.
        """
        # B
        bsize, _ = wavs.shape
        # [B, F, T / S], complex64
        fft = torch.stft(
            wavs,
            self.config.mel_win,
            self.config.mel_hop,
            self.config.mel_win,
            self.window,
            return_complex=True)
        # PEQ
        if quality_power is not None:
            # alias
            q_min, q_max = self.config.q_min, self.config.q_max
            # [B, num_peak + 2]
            q = q_min * (q_max / q_min) ** quality_power
            if gain is None:
                # [B, num_peak]
                gain = torch.zeros_like(q[:, :-2])
            # [B, num_peak]
            center = self.peak_centers[None].repeat(bsize, 1)
            # [B, F]
            peaks = torch.prod(
                self.peq.peaking_equalizer(center, gain[:, :-2], q[:, :-2]), dim=1)
            # [B, F]
            lowpass = self.peq.low_shelving(
                self.config.cutoff_lowpass, gain[:, -2], q[:, -2])
            highpass = self.peq.high_shelving(
                self.config.cutoff_highpass, gain[:, -1], q[:, -1])
            # [B, F]
            filters = peaks * highpass * lowpass
            # [B, F, T / S]
            fft = fft * filters[..., None]
        # [B, T]
        out = torch.istft(
            fft,
            self.config.mel_win,
            self.config.mel_hop,
            self.config.mel_win,
            self.window).clamp(-1., 1.)
        # max value normalization
        out = out / out.abs().amax(dim=-1, keepdim=True).clamp_min(1e-7)
        if formant_shift is None and pitch_shift is None and pitch_range is None:
            return out
        # praat-based augmentation
        if formant_shift is None:
            formant_shift = torch.ones(bsize)
        if pitch_shift is None:
            pitch_shift = torch.ones(bsize)
        if pitch_range is None:
            pitch_range = torch.ones(bsize)
        out = torch.tensor(
            np.stack([
                self.praat.augment(o, fs.item(), ps.item(), pr.item())
                for o, fs, ps, pr in zip(
                    out.cpu().numpy(),
                    formant_shift.cpu().numpy(),
                    pitch_shift.cpu().numpy(),
                    pitch_range.cpu().numpy())], axis=0),
            device=out.device, dtype=torch.float32)
        return out
