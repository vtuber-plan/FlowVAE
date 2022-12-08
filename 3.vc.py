import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import torch
import utils
from models import SynthesizerTrn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model():
    hps = utils.get_hparams_from_file("./configs/vctk_hubert.json")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint("logs/cn/G_300000.pth", net_g, None)

    return net_g

def encode_dataset(args):
    net_g = load_model()

    sid = np.load(args.sid)
    sid = torch.FloatTensor(sid).cuda()
    sid.unsqueeze_(0)

    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        sunit = np.load(in_path)
        sunit = torch.FloatTensor(sunit).cuda()
        sunit.unsqueeze_(0)
        sunit_length = torch.LongTensor([sunit.size(1)]).cuda()
        
        w = net_g.infer(sunit.transpose(1,2), sunit_length, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(out_path).replace(".npy",".wav")
        sf.write(out_path,w,16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "sid",
        metavar="sid",
        type=Path,
    )
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
        help="extension of the audio files (defaults to .npy).",
        default=".npy",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
