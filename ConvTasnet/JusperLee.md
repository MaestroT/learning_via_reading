https://github.com/JusperLee/Conv-TasNet

数据加载方式：
首先创建scp文件，记录训练数据的文件名和路径：
```
<file_name><space><file_path>
```
音频加载脚本：
AudioReader.py
```python
import torchaudio
import torch
from utils import handle_scp

# 将scp文件的信息存储为dict，filename和filepath一一对应
def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'} # {'filename': 'filepath'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict

# 读取一条音频文件
def read_wav(fname, return_rate=False):
    '''
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L 
                     L is the number of audio frames 
                     C is the number of channels. 
                sr: sample rate
    '''
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()


def write_wav(fname, src, sample_rate):
    '''
         Write wav file
         input:
               fname: wav file path
               src: frames of audio
               sample_rate: An integer which is the sample rate of the audio
         output:
               None
    '''
    torchaudio.save(fname, src, sample_rate)


class AudioReader(object):
    '''
        Class that reads Wav format files
        Input as a different scp file address
        Output a matrix of wav files in all scp files.
    '''

    def __init__(self, scp_path, sample_rate=8000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys()) # 存储所有音频文件名的list

    def _load(self, key):
        src, sr = read_wav(self.index_dict[key], return_rate=True) # 读取一条音频文件
        if self.sample_rate is not None and sr != self.sample_rate:
            raise RuntimeError('SampleRate mismatch: {:d} vs {:d}'.format(
                sr, self.sample_rate))
        return src

    def __len__(self):
        return len(self.keys) # 返回音频文件个数

    def __iter__(self):
        for key in self.keys:
            yield key, self._load(key) # 迭代生成 音频文件名 和 对应的音频数据

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError('Unsupported index type: {}'.format(type(index)))
        if type(index) == int:
            num_uttrs = len(self.keys)
            if num_uttrs < index and index < 0:
                raise KeyError('Interger index out of range, {:d} vs {:d}'.format(
                    index, num_uttrs))
            index = self.keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))

        return self._load(index) # 返回一条音频文件


if __name__ == "__main__":
    r = AudioReader('/home/likai/data1/create_scp/cv_s2.scp')
    index = 0
    print(r[1])
```


DataLoaders.py:
```python
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from AudioReader import AudioReader
import torch.nn.functional as F
import random


def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    dataset = Datasets(**data_kwargs)
    return DataLoaders(dataset,
                      is_train=is_train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


# Dataset 类，分别对 mixture 和 ref 调用 AudioReader 加载音频
class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, mix_scp=None, ref_scp=None, sr=8000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(mix_scp, sample_rate=sr)
        self.ref_audio = [AudioReader(r, sample_rate=sr) for r in ref_scp]

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        key = self.mix_audio.keys[index]
        mix = self.mix_audio[key]
        ref = [r[key] for r in self.ref_audio]
        return {
            'mix': mix,
            'ref': ref
        }


class Spliter():
    '''
       Split the audio. All audio is divided 
       into 4s according to the requirements in the paper.
       input:
             chunk_size: split size = sample_rate * 4s
             least: Less than this value will not be read
    '''

    def __init__(self, chunk_size=32000, is_train=True, least=16000):
        super(Spliter, self).__init__()
        self.chunk_size = chunk_size
        self.is_train = is_train
        self.least = least

    def chunk_audio(self, sample, start):
        '''
           Make a chunk audio
           sample: a audio sample
           start: split start time
        '''
        chunk = dict()
        chunk['mix'] = sample['mix'][start:start+self.chunk_size]
        chunk['ref'] = [r[start:start+self.chunk_size] for r in sample['ref']]
        return chunk

    def splits(self, sample):
        '''
           Split a audio sample
           小于最小长度：return []
           小于chunk_size: 补零
           大于chunk_size: random
        '''
        length = sample['mix'].shape[0]
        if length < self.least:
            return []
        audio_lists = []
        if length < self.chunk_size:
            gap = self.chunk_size-length
            sample['mix'] = F.pad(sample['mix'], (0, gap), mode='constant')
            sample['ref'] = [F.pad(r, (0, gap), mode='constant')
                             for r in sample['ref']]
            audio_lists.append(sample)
        else:
            # random, 按照least统计
            random_start = random.randint(
                0, length % self.least) if self.is_train else 0
            while True:
                if random_start+self.chunk_size > length:
                    break
                audio_lists.append(self.chunk_audio(sample, random_start))
                random_start += self.least
        return audio_lists


class DataLoaders():
    '''
        Custom dataloader method
        input:
              dataset (Dataset): dataset from which to load the data.
              num_workers (int, optional): how many subprocesses to use for data (default: 4)
              chunk_size (int, optional): split audio size (default: 32000(4 s))
              batch_size (int, optional): how many samples per batch to load
              is_train: if this dataloader for training
    '''

    def __init__(self, dataset, num_workers=4, chunk_size=32000, batch_size=1, is_train=True):
        super(DataLoaders, self).__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.is_train = is_train
        self.data_loader = DataLoader(self.dataset,
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size // 2,
                                      shuffle=self.is_train,
                                      collate_fn=self._collate)
        self.spliter = Spliter(
            chunk_size=self.chunk_size, is_train=self.is_train, least=self.chunk_size // 2)

    def _collate(self, batch):
        '''
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        '''
        batch_audio = []
        for b in batch:
            batch_audio += self.spliter.splits(b)
        return batch_audio

    def __iter__(self):
        mini_batch = []
        for batch in self.data_loader:
            mini_batch += batch
            length = len(mini_batch)
            if self.is_train:
                random.shuffle(mini_batch)
            collate_chunk = []
            for start in range(0, length-self.batch_size+1, self.batch_size):
                b = default_collate(
                    mini_batch[start:start+self.batch_size])
                collate_chunk.append(b)
            idx = length % self.batch_size
            mini_batch = mini_batch[-idx:] if idx else []
            for m_batch in collate_chunk:
                yield m_batch # batch of datasets
                '''
                   mini_batch like this
                   'mix': batch x L
                   'ref': [bathc x L, bathc x L]
                '''


if __name__ == "__main__":
    datasets = Datasets('/home/likai/data1/create_scp/cv_mix.scp',
                        ['/home/likai/data1/create_scp/cv_s1.scp', '/home/likai/data1/create_scp/cv_s2.scp'])
    dataloaders = DataLoaders(datasets, num_workers=0,
                              batch_size=10, is_train=False)
    for eg in dataloaders:
        print(eg)
        import pdb
        pdb.set_trace()
```