import torch
import configparser
from pathlib import Path
from transformers import BertConfig, BertTokenizer, BertModel
from utils import create_dataset


class ConllDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.exists()
        self.fname_list, self.conll_path = self.integrate_files()
        self.dataset, self.label_list = create_dataset(self.conll_path)

    def integrate_files(self):
        fname_list, conll_list = [], []

        ### Abstract level ###
        for conll_p in self.dataset_dir.glob("*.conll"):
            if "all" not in conll_p.name:
                fname_list.append(conll_p.stem)
                with conll_p.open("r") as f:
                    text = f.read()
                    text = text.replace("\n\n", "\n")
                    conll_list.append(text)
        save_path = self.dataset_dir / "all.conll"

        ### Sentence level ###
        # for conll_p in self.dataset_dir.glob("*.conll"):
        #     if "all" not in conll_p.name:
        #         with conll_p.open("r") as f:
        #             text = f.read()
        #             sents = text.split("\n\n")
        #             fname_list.extend([conll_p.stem + f"_{i}_{len(sents)}" for i in range(1, len(sents) + 1)])
        #             conll_list.extend(sents)
        # save_path = self.dataset_dir / "sent_all.conll"

        with save_path.open("w") as f:
            f.write("\n\n".join(conll_list))
        return fname_list, save_path

    def get_position(self):
        return [int(fname.split("_")[1]) for fname in self.fname_list]

    def transforms(self):
        pass

    def __getitem__(self, i):
        return self.fname_list[i], self.dataset[i]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset_dir = "/data/MainCLF1000/sentences/cross_validation/5_fold/train1/"
    conll_dataset = ConllDataset(dataset_dir)
    import pdb; pdb.set_trace()
