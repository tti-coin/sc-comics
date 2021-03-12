import argparse
from pathlib import Path
from copy import copy


def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dygiepp_dir", type=str, help="")
    parser.add_argument("mmi_dir", type=str, help="")
    parser.add_argument("save_dir", type=str, help="")
    return parser.parse_args()


def read_ann(ann_p):
    parsed_entities = []
    others = []
    with ann_p.open() as f:
        lines = [l.strip() for l in f.readlines()]
    for l in lines:
        if l.startswith("T"):
            tid, info, text = l.split("\t")
            label, start, end = info.split()
            parsed_entities.append([tid, label, start, end, text])
        else:
            others.append(l)
    return parsed_entities, others


def save_ann(save_p, lines):
    with save_p.open("w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    args = get_args()
    dygiepp_dir = Path(args.dygiepp_dir)
    mmi_dir = Path(args.mmi_dir)
    # save_dir = Path("../out/merged/dev1_val_2_test_1/")
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    for d_ann_p in dygiepp_dir.glob("*.ann"):
        d_parsed_entities, others = read_ann(d_ann_p)
        m_ann_p = mmi_dir / d_ann_p.name
        m_parsed_entities, _ = read_ann(m_ann_p)
        parsed_entities = []
        for dpe in d_parsed_entities:
            if dpe[1] == "Element":
                for mpe in m_parsed_entities:
                    if dpe[2:4] == mpe[2:4]:
                        dpe_copy = copy(dpe)
                        dpe_copy[1] = mpe[1]
                        parsed_entities.append(dpe_copy)
                        break
                else:
                    parsed_entities.append(dpe)
            else:
                parsed_entities.append(dpe)
        entities = []
        for pe in parsed_entities:
            tid, label, start, end, text = pe
            entities.append(f"{tid}\t{label} {start} {end}\t{text}")
        save_p = save_dir / d_ann_p.name
        lines = entities + others
        save_ann(save_p, lines)
