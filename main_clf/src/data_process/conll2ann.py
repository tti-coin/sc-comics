import argparse
from pathlib import Path
import subprocess


source = "../../../brat_tool/BIOtoStandoff.py"
parser = argparse.ArgumentParser()
parser.add_argument("conll_dir", type=str, help="")
parser.add_argument("text_dir", type=str, help="")
args = parser.parse_args()
conll_dir = args.conll_dir
text_dir = args.text_dir

for conll_path in Path(conll_dir).glob("*.conll"):
    basename = conll_path.stem
    text_path = str(Path(text_dir) / (basename + ".txt"))
    save_path = str(Path(conll_dir) / (basename + ".ann"))
    conll_path = str(conll_path)
    subprocess.run(f"cp {text_path} {conll_dir}", shell=True, executable="/bin/bash")

    subprocess.run(f"python3 {source} {text_path} {conll_path} > {save_path}", shell=True, executable="/bin/bash")
