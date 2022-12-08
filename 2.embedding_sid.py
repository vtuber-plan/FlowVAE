import argparse
import os
import pathlib
from pathlib import Path
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from tqdm import tqdm
from torchaudio.functional import resample

def filepath_generator(path, format='.wav'):
    p = pathlib.Path(path)
    for f in p.rglob('*'+format):
        yield f

def encode_dataset(args):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
    g = filepath_generator(args.in_dir, args.extension)
    os.makedirs(args.out_dir, exist_ok=True)
    for p in tqdm(list(g)):
        signal,sr = torchaudio.load(p)
        if sr != 16000:
            signal = resample(signal, sr, 16000)
        n = classifier.encode_batch(signal).squeeze(0).squeeze(0).detach().cpu().numpy()
        np.save(args.out_dir / p.name.replace(".wav",".npy"), n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .wav).",
        default=".wav",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)