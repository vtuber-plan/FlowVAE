import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import torch
import utils
from models import SynthesizerTrn
import os
import yaml
from utils import load_wav_to_torch
from mel_processing import spectrogram_torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model(hps,ckpt):
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(ckpt, net_g, None)

    return net_g

def get_spec(path, hps):
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate !=  hps.data.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, hps.data.sampling_rate))
    audio_norm = audio / 32768
    audio_norm = audio_norm.unsqueeze(0)

    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False).cuda()
    length = torch.LongTensor([spec.size(-1)]).cuda()
    # spec = torch.squeeze(spec, 0)
    return spec, length

def main(args):
    hps = utils.get_hparams_from_file(args.config)
    net_g = load_model(hps, args.ckpt)

    data = yaml.load(open(args.data, 'r'), Loader=yaml.FullLoader)
    for k,v in tqdm(data.items()):
        sid = v['target'][0].split('/')[-1].split('.')[0]
        sid = np.load("data/sid_ecape_tdnn_192dim/{}.npy".format(sid))
        sid = torch.FloatTensor(sid).cuda().unsqueeze_(0)
        
        spec, length = get_spec(v['source'], hps)
        audio = net_g.infer(spec, length, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

        out_path = args.out_dir / (k + '.wav')
        os.makedirs(out_path.parent, exist_ok=True)
        sf.write(out_path, audio, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "config",
        metavar="config",
        type=Path,
        default="./configs/vctk_e2e_32.json",
    )
    parser.add_argument(
        "ckpt",
        metavar="ckpt",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "data",
        metavar="data",
        default="./configs/vc_rule.yaml",
        type=Path,
    )
    args = parser.parse_args()
    main(args)
