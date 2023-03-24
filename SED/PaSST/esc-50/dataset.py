import io
import os
import pathlib
import random

import av
import librosa
import torchaudio
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DistributedSampler, WeightedRandomSampler

import torch
from ba3l.ingredients.datasets import Dataset
import pandas as pd
from sacred.config import DynamicIngredient, CMD
from scipy.signal import convolve
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import h5py
from helpers.audiodatasets import  PreprocessDataset


LMODE = os.environ.get("LMODE", False)

dataset = Dataset('Esc50')


@dataset.config
def default_config():
    name = 'esc50'  # dataset name
    normalize = False  # normalize dataset
    subsample = False  # subsample squares from the dataset
    roll = True  # apply roll augmentation
    fold = 1
    base_dir = "audioset_hdf5s/esc50/"  # base directory of the dataset as downloaded
    if LMODE:
        base_dir = "/system/user/publicdata/CP/audioset/audioset_hdf5s/esc50/"
    meta_csv = base_dir + "meta/esc50.csv"
    audio_path = base_dir + "audio_32k/"
    ir_path = base_dir + "irs/"
    num_of_classes = 50





def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    # print(stream)
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    else:
        return x[0: audio_length]


irs_arr = None


'''
get_ir_sample is used to apply impulse response (IR) augmentation to the audio data. IR augmentation is a technique used to simulate the effect of different acoustic environments on the audio data. The function loads a set of IR files from a specified directory and randomly selects one of them to apply to the audio data. The IR file is convolved with the audio data to simulate the effect of the acoustic environment. The function returns the convolved audio data.
The IR files are stored in the .wav format and are located in the directory specified by the ir_path variable. The get_ir_sample function loads the IR files from this directory and randomly selects one of them to apply to the audio data.
'''
@dataset.command
def get_ir_sample(ir_path, _run, ir_augment, cut_irs_offset=None):
    if not ir_augment:
        return
    global irs_arr
    if irs_arr is None:
        all_paths = [path for path in pathlib.Path(os.path.expanduser(ir_path)).rglob('*.wav')]
        all_paths = sorted(all_paths)
        if cut_irs_offset is not None:
            all_paths = all_paths[cut_irs_offset:cut_irs_offset + 10]
        all_paths_name = [str(p).rsplit("/", 1)[-1] for p in all_paths]
        print("will use these IRs:")
        for i in range(len(all_paths_name)):
            print(i, ": ", all_paths_name[i])
        _run.info["ir_devices"] = all_paths_name
        irs_arr = [librosa.load(p, sr=32000)[0] for p in all_paths]
    return irs_arr[int(np.random.randint(0, len(irs_arr)))]


'''
The gain_augment parameter in the pydub_augment function controls the amount of gain augmentation to apply to the audio data. If gain_augment is set to a non-zero value, the function will randomly adjust the amplitude of the audio data by a factor of gain_augment * 2 decibels. For example, if gain_augment is set to 7, the amplitude of the audio data will be randomly adjusted by a factor of up to 14 decibels.
'''
@dataset.command
def pydub_augment(waveform, gain_augment=7, ir_augment=0):
    if ir_augment and torch.rand(1) < ir_augment:
        ir = get_ir_sample()
        waveform = convolve(waveform, ir, 'full')
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        amp = 10 ** (gain / 20)
        waveform = waveform * amp
    return waveform


'''
The MixupDataset class is used to mix up waveforms. It takes a dataset as input and returns a new dataset where each audio clip is a mix of two randomly selected audio clips from the original dataset. The mixing is done by taking a weighted average of the two audio clips, where the weights are randomly generated using a beta distribution with a parameter of beta. The MixupDataset class is used to augment the training data and improve the performance of the model.

In the MixupDataset class, x1 is the waveform of the first audio clip, f1 is the filename of the first audio clip, and y1 is the target label of the first audio clip. Similarly, x2, f2, and y2 are the waveform, filename, and target label of the second audio clip. The __getitem__ method of the MixupDataset class returns a mix of the two audio clips, where the mixing is done by taking a weighted average of the two waveforms.
'''
class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, f1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, f2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, f1, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)



'''
The AudioSetDataset class is used to load audio clips from the ESC-50 dataset. It takes as input the path to a CSV file containing metadata about the audio clips, the path to the directory containing the audio files, the fold number to use for training or testing, and various other parameters. The __getitem__ method of the AudioSetDataset class returns a tuple containing the waveform of the audio clip, the filename of the audio clip, and the target label of the audio clip. The waveform is loaded from the audio file using the librosa.load function, and is then padded or truncated to a fixed length. If the augment parameter is set to True, the waveform is then passed through the pydub_augment function to apply various types of data augmentation.m
'''
class AudioSetDataset(TorchDataset):
    def __init__(self, meta_csv,  audiopath, fold, train=False, sample_rate=32000, classes_num=527,
                 clip_length=5, augment=False):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.sample_rate = sample_rate
        self.meta_csv = meta_csv
        self.df = pd.read_csv(meta_csv)
        if train:  # training all except this
            print(f"Dataset training fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold != fold]
            print(f" for training remains {len(self.df)}")
        else:
            print(f"Dataset testing fold {fold} selection out of {len(self.df)}")
            self.df = self.df[self.df.fold == fold]
            print(f" for testing remains {len(self.df)}")

        self.clip_length = clip_length * sample_rate
        self.sr = sample_rate
        self.classes_num = classes_num
        self.augment = augment
        self.audiopath=audiopath
        if augment:
            print(f"Will agument data from {meta_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]

        #waveform = decode_mp3(np.fromfile(self.audiopath + row.filename, dtype='uint8'))
        waveform, _ = librosa.load(self.audiopath + row.filename, sr=self.sr, mono=True)
        if self.augment:
            waveform = pydub_augment(waveform)
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = self.resample(waveform)
        target = row.target
        return waveform.reshape(1, -1),  row.filename, target

    '''
    The resample function in the AudioSetDataset class is used to 
    resample an audio clip to a fixed length. It takes as input the 
    waveform of the audio clip, and returns the waveform padded or 
    truncated to a fixed length.
    '''
    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0:: 2]
        elif self.sample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')



@dataset.command
def get_base_training_set(meta_csv, audio_path, fold=1):
    ds = AudioSetDataset(meta_csv, audio_path, fold,  train=True, augment=True)
    return ds


@dataset.command
def get_ft_weighted_sampler(samples_weights=CMD(".get_ft_cls_balanced_sample_weights"),
                            epoch_len=100000, sampler_replace=False):
    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    print("num_nodes= ", num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    return DistributedSamplerWrapper(sampler=WeightedRandomSampler(samples_weights,
                                                                   num_samples=epoch_len, replacement=sampler_replace),
                                     dataset=range(epoch_len),
                                     num_replicas=num_nodes,
                                     rank=rank,
                                     )


@dataset.command
def get_base_test_set(meta_csv, audio_path, fold=1):
    ds = AudioSetDataset(meta_csv, audio_path, fold,  train=False)
    return ds


'''
The get_roll_func function returns a function that can be used to 
apply a random roll augmentation to the audio data. The function 
takes as input the axis along which to roll the audio data, the 
amount by which to shift the audio data (if None, a random shift 
is chosen), and the range of possible shift values. The function 
returns a new function that takes as input a batch of audio data, 
and applies the roll augmentation to each audio clip in the batch. 
The roll augmentation is applied by calling the roll method of the 
PyTorch tensor object. The get_roll_func function is used in the 
get_training_set function to apply roll augmentation to the training data.
'''
@dataset.command(prefix='roll_conf')
def get_roll_func(axis=1, shift=None, shift_range=50):
    print("rolling...")

    def roll_func(b):
        x, i, y = b
        x = torch.as_tensor(x)
        sf = shift
        if shift is None:
            sf = int(np.random.random_integers(-shift_range, shift_range))
        global FirstTime

        return x.roll(sf, axis), i, y

    return roll_func


@dataset.command
def get_training_set(normalize, roll, wavmix=False):
    ds = get_base_training_set()
    get_ir_sample()
    if normalize:
        print("normalized train!")
        fill_norms()
        ds = PreprocessDataset(ds, norm_func)
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)

    return ds


@dataset.command
def get_test_set(normalize):
    ds = get_base_test_set()
    if normalize:
        print("normalized test!")
        fill_norms()
        ds = PreprocessDataset(ds, norm_func)
    return ds


@dataset.command
def print_conf(_config):
    print("Config of ", dataset.path, id(dataset))
    print(_config)
    print()


'''
The DistributedSamplerWrapper class is a wrapper around the DistributedSampler class in PyTorch. 
It is used to distribute the data across multiple nodes in a distributed training setup. 
The DistributedSamplerWrapper class takes as input a sampler object and a dataset object, 
and returns a new sampler object that can be used to distribute the data across multiple nodes. 
The DistributedSamplerWrapper class overrides the __iter__ method of the DistributedSampler class 
to ensure that the data is distributed evenly across all nodes. The DistributedSamplerWrapper class 
is used in the get_ft_weighted_sampler function to create a weighted random sampler for the training data.
'''
class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


if __name__ == "__main__":
    from sacred import Experiment

    ex = Experiment("test_dataset", ingredients=[dataset])


    @ex.automain
    def default_command():
        ex.current_run.get_command_function("print_config")()
        get_base_training_set()
        ds = get_test_set()
        print(ds[0])
        ds = get_training_set()
        print(ds[0])
        print("get_base_training_set", len(get_base_training_set()))
        print("get_base_test_set", len(get_base_test_set()))
        print("get_training_set", len(get_training_set()))
        print("get_test_set", len(get_test_set()))