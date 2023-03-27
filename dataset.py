import os
import random
import torch
import commons
import torch.utils.data
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
import numpy as np
from augment import Data


def load_filepaths(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


"""Multi speaker version"""


class AudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_unit, hparams):
        self.audiopaths_sid_unit = load_filepaths(audiopaths_sid_unit)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        # self.sampling_rate = hparams.sampling_rate

        self.min_unit_len = getattr(hparams, "min_unit_len", 1)
        self.max_unit_len = getattr(hparams, "max_unit_len", 1000)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_unit)

    def get_unit_audio_speaker_pair(self, audiopath_sid_unit):
        # separate filename, speaker_id and content vector
        audiopath, sid = audiopath_sid_unit[0], audiopath_sid_unit[1]

        # unit = self.get_unit(unitpath)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        # return (unit, spec, wav, sid)
        return (spec, wav, sid)

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

    def get_unit(self, path):
        unit = np.load(path)
        unit = torch.FloatTensor(unit)
        return unit.transpose(0, 1)  # [T, C] -> [C, T]

    def get_sid(self, path):
        sid = np.load(path)
        sid = torch.FloatTensor(sid)
        return sid

    def __getitem__(self, index):
        return self.get_unit_audio_speaker_pair(self.audiopaths_sid_unit[index])

    def __len__(self):
        return len(self.audiopaths_sid_unit)


aug = Data()


class AudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, hparams, return_ids=False):
        self.return_ids = return_ids
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

    def __call__(self, batch):
        """Collate's training batch from unit, audio and speaker identities
        PARAMS
        ------
        batch: [unit_normalized, spec_normalized, wav_normalized, sid]
        """

        # Sort batch by spec length (descending order)
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)

        # max_unit_len = max([x[0].size(1) for x in batch])
        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])

        # unit_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        # unit_padded = torch.FloatTensor(
        #     len(batch), batch[0][0].size(0), max_unit_len)
        spec_padded = torch.FloatTensor(
            len(batch), batch[0][0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        sid = torch.FloatTensor(len(batch), batch[0][2].size(0))
        # unit_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # unit = row[0]
            # unit_padded[i, :, :unit.size(1)] = unit
            # unit_lengths[i] = unit.size(1)

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[1]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i, :] = row[2]

        global aug
        wav_aug = aug.augment(wav_padded.squeeze(1))
        spec_aug = spectrogram_torch(wav_aug, self.filter_length, self.sampling_rate,
                                     self.hop_length, self.win_length, center=False)

        # spec_aug is padded, lengths same as spec_padded.
        if self.return_ids:
            return spec_aug, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
            # return unit_padded, unit_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing, spec_aug
        return spec_aug, spec_padded, spec_lengths, wav_padded, wav_lengths, sid
