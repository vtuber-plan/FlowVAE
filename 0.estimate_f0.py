import argparse
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
from multiprocessing import Pool

def filepath_generator(path, format='.wav'):
    p = Path(path)
    for f in p.rglob('*'+format):
        yield f

# [2, t]
# f0 [t]
# uv [t]
def pyin(wavfile_path):
    sampling_rate = 16000
    wav, sr = sf.read(wavfile_path)
    if len(wav) < sr:
        return None
    if sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    wav = np.pad(wav, (int((512-256)/2), int((512-256)/2)), mode='reflect')
    f0, uv, _ = librosa.pyin(wav, sr=sr, frame_length=512, hop_length=256, center=False, pad_mode='reflect', fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.where(np.isnan(f0), 0.0, f0)
    uv = uv.astype(np.int32)
    # return np.array([f0, uv]).astype(np.float32)
    n = np.array([f0, uv]).astype(np.float32)
    np.save(args.out_dir / wavfile_path.name.replace(".wav",".npy"), n)

def encode_dataset(args):
    g = filepath_generator(args.in_dir, args.extension)
    os.makedirs(args.out_dir, exist_ok=True)
    p = list(g)
    with Pool(47) as pool:
        res = list(tqdm(pool.imap(pyin, p), total=len(p)))
        pool.close()
        pool.join()
    print('All subprocesses done.')
        # n = pyin(p,16000)
        # np.save(args.out_dir / p.name.replace(".wav",".npy"), n)


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