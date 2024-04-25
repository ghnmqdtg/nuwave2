import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTMag(nn.Module):
    def __init__(self, nfft=1024, hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(nfft), False)

    # x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        self.window = self.window.to(x.device)
        stft = torch.stft(
            x, self.nfft, self.hop, window=self.window, return_complex=True
        )  # [B, F, TT]
        #   return_complex=False)  #[B, F, TT,2]
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2))
        return mag
