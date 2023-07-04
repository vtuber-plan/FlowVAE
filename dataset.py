import os
import random
import torch
import torch.utils.data
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch, spec_to_mel_torch
from augment import Augmentor


def load_filepaths(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split(split)[0] for line in f]
    return filepaths


"""Multi speaker version"""
class AudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio
        2) computes spectrograms from audio files.
    """

    def __init__(self, labels, hparams):
        self.audiopaths = load_filepaths(labels)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.n_mel = hparams.n_mel_channels
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax

        self.min_unit_len = getattr(hparams, "min_unit_len", 1)
        self.max_unit_len = getattr(hparams, "max_unit_len", 1000)

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_data(self, audiopath):
        spec, wav = self.get_audio(audiopath)
        mel = spec_to_mel_torch(spec, self.filter_length, 
                                self.n_mel, self.sampling_rate,
                                self.mel_fmin, self.mel_fmax)
        return (spec, wav, mel)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm  # spec [c, t]

    def __getitem__(self, index):
        return self.get_data(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)


augmentor = Augmentor()


class AudioCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, hparams, return_ids=False):
        self.return_ids = return_ids
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

    def __call__(self, batch):
        """Collate's training batch from audio
        PARAMS
        ------
        batch: [spec_normalized, wav_normalized, mel_normalized]
        """

        # Sort batch by spec length (descending order)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(
            len(batch), batch[0][0].size(0), max_spec_len)
        mel_padded = torch.FloatTensor(
            len(batch), batch[0][2].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        mel_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            mel = row[2]
            mel_padded[i, :, :spec.size(1)] = mel

            wav = row[1]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        global augmentor
        wav_aug = augmentor.augment(wav_padded.squeeze(1))
        spec_aug = spectrogram_torch(wav_aug, self.filter_length, self.sampling_rate,
                                     self.hop_length, self.win_length, center=False)

        # spec_aug is padded, lengths same as spec_padded.
        if self.return_ids:
            return spec_aug, spec_padded, mel_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return spec_aug, spec_padded, mel_padded, spec_lengths, wav_padded, wav_lengths
