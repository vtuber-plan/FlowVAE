from pathlib import Path

p = Path("/data/public/dataset/aishell3_16k/f0_uv/")

def label_format(i:Path):
    tmp = i.name.replace(".npy",".wav")
    return "/data/public/dataset/aishell-3-16k/%s|%s|%s\n" %(tmp, str(i).replace("f0_uv","sid_ecapa_tdnn"), str(i).replace("f0_uv","hubert_soft"))

with open("label/train.txt", "w") as f:
    with open("label/val.txt", "w") as v:
        n = 0
        for i in p.rglob("*.npy"):
            n += 1
            if n <= 256:
                v.write(label_format(i))
            else:
                f.write(label_format(i))