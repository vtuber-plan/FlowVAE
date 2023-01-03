## Demo
[paper page](https://blog.frostmiku.com/Flow-VAE-VC/)

## Setup
1. resample the dataset sr to 16k
2. run `python 1.extract_hubert_soft.py soft /path/to/dataset data/hubert16k` to extract unit
3. run `python 2.embedding_sid.py /path/to/dataset data/spkerEmbed` to extract sid
4. set train/val labels into the label dir, format as `wav_path|sid_path|uint_path`
5. run `python train.py -m YourModelName` to trian a vc model

## VC

- run `python 3.vc /path/to/sid /path/to/source_dir /path/to/output_dir`
