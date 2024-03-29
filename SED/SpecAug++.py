import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Zero-value masking (ZM):
drops random stripes of a specified width and number along 
either the time or frequency dimension of the input spectrogram. 
'''
class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input


    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]    # t, f    masking width
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]  # t0, f0  masking start point

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


'''
adds Gaussian noise to the input spectrogram.
'''
class AddNoise(nn.Module):
    def __init__(self, std=1e-4):
        """Add Gaussian noise.
        Args:
          snr: float
        """
        super(AddNoise, self).__init__()

        self.std = std

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            input = input + (self.std**0.5)*torch.randn(input.shape[0],input.shape[1],input.shape[2],input.shape[3]).cuda()

            return input


'''
Zero-value masking (ZM): (Normal SpecAug)
'''
class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width,
        freq_stripes_num, add_=True, std=1e-2):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
          std: float
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width,
            stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width,
            stripes_num=freq_stripes_num)

        self.add_noise = AddNoise(std=std)
        self.add_ = add_
    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        if self.add_:
            x = self.add_noise(x)
        return x


'''
Mini-batch based mixture masking (MM):
mixes random stripes of a specified width and number 
from a randomly selected input spectrogram with the input spectrogram. 
'''
class MixStripes(nn.Module):
    def __init__(self, dim, mix_width, stripes_num):
        """Mix stripes.
        Args:
          dim: int, dimension along which to mix
          mix_width: int, maximum width of stripes to mix
          stripes_num: int, how many stripes to mix
        """
        super(MixStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.mix_width = mix_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]
            index = torch.randperm(batch_size).cuda()
            rand_ = input[index, :]
            for n in range(batch_size):
                self.transform_slice(input[n], rand_[n], total_width)

            return input


    def transform_slice(self, e, r, total_width):
        """e: (channels, time_steps, freq_bins)"""
        """r: (channels, time_steps, freq_bins)"""
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.mix_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = .5*e[:, bgn : bgn + distance, :] + .5*r[:, bgn : bgn + distance, :]
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = .5*e[:, :, bgn : bgn + distance] + .5*r[:, :, bgn : bgn + distance]


'''
Mini-batch based mixture masking (MM):
'''
class SpecMixAugmentation(nn.Module):
    def __init__(self, time_mix_width, time_stripes_num, freq_mix_width,
        freq_stripes_num):
        """Mix augmetation.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
          std: float
        """

        super(SpecMixAugmentation, self).__init__()

        self.time_mixer = MixStripes(dim=2, mix_width=time_mix_width,
            stripes_num=time_stripes_num)

        self.freq_mixer = MixStripes(dim=3, mix_width=freq_mix_width,
            stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_mixer(input)
        x = self.freq_mixer(x)

        return x


'''
Mini-batch based cutting masking (CM):
cuts random stripes of a specified width and number along 
either the time or frequency dimension of the input spectrogram.
'''
class CutStripes(nn.Module):
    def __init__(self, dim, cut_width, stripes_num):
        """Cut stripes.
        Args:
          dim: int, dimension along which to cut
          cut_width: int, maximum width of stripes to cut
          stripes_num: int, how many stripes to cut
        """
        super(CutStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.cut_width = cut_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]
            index = torch.randperm(batch_size).cuda()
            rand_ = input[index, :]
            for n in range(batch_size):
                self.transform_slice(input[n], rand_[n], total_width)

            return input

    def transform_slice(self, e, r, total_width):
        """e: (channels, time_steps, freq_bins)"""
        """r: (channels, time_steps, freq_bins)"""
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.cut_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = r[:, bgn : bgn + distance, :]
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = r[:, :, bgn : bgn + distance]


'''
Mini-batch based cutting masking (CM):
'''
class SpecCutAugmentation(nn.Module):
    def __init__(self, time_cut_width, time_stripes_num, freq_cut_width,
        freq_stripes_num):
        """Cut augmetation.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
          std: float
        """

        super(SpecCutAugmentation, self).__init__()

        self.time_cutter = CutStripes(dim=2, cut_width=time_cut_width,
            stripes_num=time_stripes_num)

        self.freq_cutter = CutStripes(dim=3, cut_width=freq_cut_width,
            stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_cutter(input)
        x = self.freq_cutter(x)

        return x