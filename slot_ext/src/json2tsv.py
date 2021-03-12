import argparse
from pathlib import Path
import json


def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("json_path", type=str, help="")
    parser.add_argument("tsv_path", type=str, help="")
    return parser.parse_args()


class JSON2TSV:
    def __init__(self, json_path, tsv_path):
        self.json_path = Path(json_path)
        self.tsv_path = Path(tsv_path)

    def read_json(self):
        with self.json_path.open() as f:
            df = json.load(f)
        return df

    def extract_text(self, slot):
        Element = slot["Element"]["text"]
        if slot["Element"]["comp_elems"]:
            temp = []
            for k, v in slot["Element"]["comp_elems"].items():
                temp.append(k + ":" + str(v))
            comp_elems = ",".join(temp)
        else:
            comp_elems = ""
        if slot["Tc"]:
            Tc = slot["Tc"]["text"]
        else:
            Tc = ""
        if slot["Dopant"]:
            Dopant = slot["Dopant"]["text"]
        else:
            Dopant = ""
        if slot["Site"]:
            Site = slot["Site"]["text"]
        else:
            Site = ""
        return Element, comp_elems, Tc, Dopant, Site

    def merge_slot(self, df):
        merged_df = {}
        for fname, slots in df["docs"].items():
            for slot in slots:
                slot = self.extract_text(slot)
                if slot not in merged_df:
                    merged_df[slot] = {fname}
                else:
                    merged_df[slot].add(fname)
        return merged_df

    def convert(self):
        df = self.read_json()
        merged_df = self.merge_slot(df)
        lines = ["Element\tcomp_elems\tTc\tDopant\tSite\tdoc_id"]
        for slot, fname_set in merged_df.items():
            line = "\t".join(slot) + "\t" + ",".join(list(fname_set))
            lines.append(line)
        with self.tsv_path.open("w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    args = get_args()
    json_path = Path(args.json_path)
    tsv_path = Path(args.tsv_path)
    JSON2TSV(json_path, tsv_path).convert()
    # elem, comp_elems, Tc, Dopant, Site, doc_id
