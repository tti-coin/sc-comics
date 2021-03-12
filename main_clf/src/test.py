import numpy as np
from tqdm import tqdm
from pathlib import Path
import configparser
import torch
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer
from dataset import ConllDataset
from model import MLP
from utils import convert_single_example, Score
from train import Trainer


class Test():
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        self.save_dir = Path(config.get("general", "save_dir"))
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.clf_th = config.getfloat("general", "clf_th")

        self.mlp_model_path = config.get("model", "mlp")
        assert Path(self.mlp_model_path).exists()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_config_path = config.get("bert", "config_path")
        assert Path(bert_config_path).exists()
        self.bert_config = LongformerConfig.from_json_file(bert_config_path)
        self.max_seq_length = self.bert_config.max_position_embeddings - 2
        self.bert_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        # bert_tokenizer_path = config.get("bert", "tokenizer_path")
        # assert Path(bert_config_path).exists()
        # self.bert_tokenizer = LongformerTokenizer.from_pretrained(bert_tokenizer_path)
        bert_model_path = config.get("bert", "model_path")
        assert Path(bert_model_path).exists()
        self.bert_model = LongformerModel.from_pretrained(bert_model_path, config=self.bert_config)
        self.bert_model.to(self.device)
        self.bert_model.eval()

        gold_dir = Path(config.get("data", "gold_dir"))
        assert Path(gold_dir).exists()
        self.gold_dataset = ConllDataset(gold_dir)
        target_dir = Path(config.get("data", "target_dir"))
        assert Path(target_dir).exists()
        self.target_dataset = ConllDataset(target_dir)

    def transforms(self, example, label_list, is_gold):
        feature = convert_single_example(example, label_list, self.max_seq_length, self.bert_tokenizer)
        label_ids = feature.label_ids
        label_map = feature.label_map
        if is_gold:
            gold_labels = [-1] * self.max_seq_length
            # Get "Element" or "Main" token indices
            for i, lid in enumerate(label_ids):
                if lid == label_map['B-Element']:
                    gold_labels[i] = 0
                elif lid == label_map['B-Main']:
                    gold_labels[i] = 1
                elif lid in (label_map['I-Element'], label_map['I-Main']):
                    gold_labels[i] = 2
                elif lid == label_map['X']:
                    gold_labels[i] = 3
            gold_labels = gold_labels
        else:
            gold_labels = [-1] * self.max_seq_length
            # Get "Element" or "Main" token indices
            for i, lid in enumerate(label_ids):
                if lid == label_map['B-Element']:
                    gold_labels[i] = 0
                elif lid  == label_map['I-Element']:
                    gold_labels[i] = 2
                elif lid == label_map['X']:
                    gold_labels[i] = 3
            gold_labels = gold_labels
        # flush data to bert model
        input_ids = torch.tensor(feature.input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            bert_output = self.bert_model(input_ids)
        # lstm (ignore padding parts)
        bert_fv = bert_output[0]
        input_ids = torch.tensor(feature.input_ids)
        label_ids = torch.tensor(feature.label_ids)
        return bert_fv, input_ids, label_ids, label_map, gold_labels

    def load_model(self):
        # MLP
        self.mlp = MLP(self.bert_config.hidden_size)
        self.mlp.load_state_dict(torch.load(self.mlp_model_path))
        self.mlp.to(self.device)
        self.mlp.eval()

    def eval(self):
        self.load_model()

        correct_save_dir = self.save_dir / "correct"
        if not correct_save_dir.exists():
            correct_save_dir.mkdir(parents=True)
        incorrect_save_dir = self.save_dir / "incorrect"
        if not incorrect_save_dir.exists():
            incorrect_save_dir.mkdir(parents=True)

        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for gold_data, target_data in tqdm(zip(self.gold_dataset, self.target_dataset)):
                # flush to Bert
                gold_fname, gold_example = gold_data
                target_fname, target_example = target_data
                if not gold_fname == target_fname:
                    import pdb; pdb.set_trace()
                assert gold_fname == target_fname

                _, _, _, _, gold_labels = self.transforms(gold_example, self.gold_dataset.label_list, is_gold=True)
                fvs, input_ids, label_ids, label_map, pred_labels = self.transforms(target_example, self.target_dataset.label_list, is_gold=False)

                # extract Element/Main tokens
                is_correct = True
                _, ent_gold_labels, golds_mask = Trainer.extract_tokens(fvs.squeeze(0), gold_labels)
                golds = {}
                if len(ent_gold_labels) >= 1:
                    i = 0
                    while True:
                        try:
                            ent_start = golds_mask.index(i)
                        except ValueError:
                            break
                        for n, j in enumerate(golds_mask[ent_start:]):
                            if j != i:
                                ent_end = (ent_start + n - 1)
                                break
                        golds[(ent_start, ent_end)] = ent_gold_labels[i]
                        i += 1

                ents, ent_pred_labels, preds_mask = Trainer.extract_tokens(fvs.squeeze(0), pred_labels)

                preds = {}
                if len(ent_pred_labels) >= 1:
                    i = 0
                    while True:
                        try:
                            ent_start = preds_mask.index(i)
                        except ValueError:
                            break
                        for n, j in enumerate(preds_mask[ent_start:]):
                            if j != i:
                                ent_end = (ent_start + n - 1)
                                break
                        preds[(ent_start, ent_end)] = ent_pred_labels[i]
                        i += 1
                for gold_span, gold_label in golds.items():
                    if gold_span not in preds.keys():
                        if gold_label == 1:
                            fn += 1
                            is_correct = False

                ents_pred = [0] * len(ents)
                for i, pred in enumerate(preds):
                    # convert to torch.tensor
                    inputs = torch.empty([len(ents[i]), self.bert_config.hidden_size]).to(self.device)
                    for j, token in enumerate(ents[i]):
                        inputs[j, :] = token

                    inputs = torch.mean(inputs, dim=0, keepdim=True)
                    outputs = self.mlp(inputs)

                    if pred in golds.keys():
                        target = golds[pred]
                        if target == 1:
                            if outputs < self.clf_th:
                                fn += 1
                                is_correct = False
                            else:
                                tp += 1
                        else:
                            if outputs < self.clf_th:
                                tn += 1
                            else:
                                fp += 1
                                is_correct = False
                    else:
                        if outputs < self.clf_th:
                            pass
                        else:
                            fp += 1
                            is_correct = False

                    outputs_ = outputs.to('cpu').detach().numpy().copy()
                    if np.all(outputs_ > self.clf_th):
                        ents_pred[i] = 1

                if is_correct:
                    save_dir = correct_save_dir
                else:
                    save_dir = incorrect_save_dir
                save_path = save_dir / (target_fname + ".conll")
                lines = []
                elem_cnt = -1
                for i in range(len(target_example.text)):
                    text = target_example.text[i]
                    label = target_example.label[i]
                    start = target_example.start[i]
                    end = target_example.end[i]
                    if label == "B-Element":
                        elem_cnt += 1
                        if ents_pred[elem_cnt] == 1:
                            lines.append(f"B-Main\t{start}\t{end}\t{text}")
                        elif ents_pred[elem_cnt] == 0:
                            lines.append(f"{label}\t{start}\t{end}\t{text}")
                    elif label == "I-Element":
                        if ents_pred[elem_cnt] == 1:
                            lines.append(f"I-Main\t{start}\t{end}\t{text}")
                        elif ents_pred[elem_cnt] == 0:
                            lines.append(f"{label}\t{start}\t{end}\t{text}")
                    else:
                        lines.append(f"{label}\t{start}\t{end}\t{text}")

                with save_path.open("w") as f:
                    f.write("\n".join(lines))

        return Score(tp, fp, tn, fn).calc_score()


if __name__ == "__main__":
    config_path = "./configs/test.conf"
    test = Test(config_path)

    acc, rec, _, prec, f1 = test.eval()
    print(f"acc: {acc}")
    print(f"rec: {rec}")
    print(f"prec: {prec}")
    print(f"f1: {f1}")
