import sys
import copy
from tqdm import tqdm
from pathlib import Path
import configparser
import torch
from torch.nn import BCEWithLogitsLoss
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer, BertConfig, LongformerModel, LongformerTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import ConllDataset
from model import MLP
from utils import convert_single_example, Score
# from adabelief_pytorch import AdaBelief


class Trainer():
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        self.n_epoch = config.getint("general", "n_epoch")
        self.batch_size = config.getint("general", "batch_size")
        self.train_bert = config.getboolean("general", "train_bert")
        self.lr = config.getfloat("general", "lr")
        self.cut_frac = config.getfloat("general", "cut_frac")
        self.log_dir = Path(config.get("general", "log_dir"))
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        self.model_save_freq = config.getint("general", "model_save_freq")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # bert_config_path = config.get("bert", "config_path")
        # bert_tokenizer_path = config.get("bert", "tokenizer_path")
        # bert_model_path = config.get("bert", "model_path")

        self.bert_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        # self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        tkzer_save_dir = self.log_dir / "tokenizer"
        if not tkzer_save_dir.exists():
            tkzer_save_dir.mkdir()
        self.bert_tokenizer.save_pretrained(tkzer_save_dir)
        self.bert_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.bert_config = self.bert_model.config
        # self.bert_config = BertConfig.from_pretrained(bert_config_path)
        # self.bert_model = BertModel.from_pretrained(bert_model_path, config=self.bert_config)
        self.max_seq_length = self.bert_config.max_position_embeddings - 2
        # self.max_seq_length = self.bert_config.max_position_embeddings
        self.bert_model.to(self.device)

        if self.train_bert:
            self.bert_model.train()
        else:
            self.bert_model.eval()

        train_conll_path = config.get("data", "train_path")
        assert Path(train_conll_path).exists()
        dev_conll_path = config.get("data", "dev_path")
        assert Path(dev_conll_path).exists()
        dev1_conll_path = Path(dev_conll_path) / "1"
        assert dev1_conll_path.exists()
        dev2_conll_path = Path(dev_conll_path) / "2"
        assert dev2_conll_path.exists()
        self.train_dataset = ConllDataset(train_conll_path)
        # self.dev_dataset = ConllDataset(dev_conll_path)
        self.dev1_dataset = ConllDataset(dev1_conll_path)
        self.dev2_dataset = ConllDataset(dev2_conll_path)
        if self.batch_size == -1:
            self.batch_size = len(self.train_dataset)

        self.scaler = torch.cuda.amp.GradScaler()
        tb_cmt = f"lr_{self.lr}_cut-frac_{self.cut_frac}"
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=tb_cmt)

    def transforms(self, example, label_list):
        feature = convert_single_example(example, label_list, self.max_seq_length, self.bert_tokenizer)
        label_ids = feature.label_ids
        label_map = feature.label_map
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
        # flush data to bert model
        input_ids = torch.tensor(feature.input_ids).unsqueeze(0).to(self.device)
        if self.train_bert:
            model_output = self.bert_model(input_ids)
        else:
            with torch.no_grad():
                model_output = self.bert_model(input_ids)

        # lstm (ignore padding parts)
        model_fv = model_output[0]
        input_ids = torch.tensor(feature.input_ids)
        label_ids = torch.tensor(feature.label_ids)
        gold_labels = torch.tensor(gold_labels)
        return model_fv, input_ids, label_ids, gold_labels

    @staticmethod
    def extract_tokens(fv, gold_labels):
        ents, golds = [], []
        ents_mask = [-1] * len(gold_labels)
        ent, gold, ent_id = [], None, 0
        ent_flag = False
        for i, gt in enumerate(gold_labels):
            if gt == 2:                 # in case of "I-xxx"
                ent.append(fv[i, :])
                ents_mask[i] = ent_id
                ent_end = i
            elif gt == 3 and ent_flag:  # in case of "X"
                ent.append(fv[i, :])
                ents_mask[i] = ent_id
                ent_end = i
            elif ent:
                ents.append(ent)
                golds.append(gold)
                ent = []
                ent_id += 1
                ent_flag = False
            if gt in (0, 1):            # in case of "B-xxx"
                ent.append(fv[i, :])
                gold = gt
                ents_mask[i] = ent_id
                ent_start = i
                ent_flag = True
        else:
            if ent:
                ents.append(ent)
                golds.append(gold)
        return ents, golds, ents_mask

    def eval(self, dataset):
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for data in tqdm(dataset):
                # flush to Bert
                fname, example = data

                try:
                    fvs, input_ids, label_ids, gold_labels = self.transforms(example, dataset.label_list)
                except RuntimeError:
                    print(f"{fname} cannot put in memory!")
                    continue

                # extract Element/Main tokens
                ents, ent_golds, _ = self.extract_tokens(fvs.squeeze(0), gold_labels)

                for i, ent in enumerate(ents):
                    # convert to torch.tensor
                    inputs = torch.empty([len(ent), self.bert_config.hidden_size]).to(self.device)
                    for j, token in enumerate(ent):
                        inputs[j, :] = token
                    target = ent_golds[i]
                    inputs = torch.mean(inputs, dim=0, keepdim=True)

                    # classification
                    outputs = self.mlp(inputs)
                    if target == 1:
                        if outputs < 0.5:
                            fn += 1
                        else:
                            tp += 1
                    else:
                        if outputs < 0.5:
                            tn += 1
                        else:
                            fp += 1

        return Score(tp, fp, tn, fn).calc_score()

    def train(self):
        # MLP
        self.mlp = MLP(self.bert_config.hidden_size)
        self.mlp.to(self.device)
        self.mlp.train()
        # learnging parameter settings
        params = list(self.mlp.parameters())
        if self.train_bert:
            params += list(self.bert_model.parameters())
        # loss
        self.criterion = BCEWithLogitsLoss()
        # optimizer
        self.optimizer = AdamW(params, lr=self.lr)
        num_train_steps = int(self.n_epoch * len(self.train_dataset) / self.batch_size)
        num_warmup_steps = int(self.cut_frac * num_train_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_train_steps)

        try:
            best_dev1_f1, best_dev2_f1 = 0, 0
            # best_dev_f1 = 0
            itr = 1
            for epoch in range(1, self.n_epoch + 1):
                print("Epoch : {}".format(epoch))
                print("training...")
                for i in tqdm(range(0, len(self.train_dataset), self.batch_size)):
                    # fvs, ents, batch_samples, inputs, outputs = None, None, None, None, None
                    itr += i
                    # create batch samples
                    if (i + self.batch_size) < len(self.train_dataset):
                        end_i = (i + self.batch_size)
                    else:
                        end_i = len(self.train_dataset)

                    batch_samples, batch_golds = [], []

                    for j in range(i, end_i):
                        # flush to Bert
                        fname, example = self.train_dataset[j]

                        fvs, input_ids, label_ids, gold_labels = self.transforms(example, self.train_dataset.label_list)

                        # extract Element/Main tokens
                        ents, ent_golds, _ = self.extract_tokens(fvs.squeeze(0), gold_labels)
                        for e in ents:
                            ent = torch.empty([len(e), self.bert_config.hidden_size]).to(self.device)
                            for k, t in enumerate(e):
                                ent[k, :] = t
                            batch_samples.append(torch.mean(ent, dim=0))
                        batch_golds.extend(ent_golds)

                    # convert to torch.tensor
                    inputs = torch.empty([len(batch_samples), self.bert_config.hidden_size]).to(self.device)
                    for j, t in enumerate(batch_samples):
                        inputs[j, :] = t
                    targets = torch.tensor(batch_golds, dtype=torch.float).unsqueeze(1)

                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = self.mlp(inputs)
                        loss = self.criterion(outputs, targets.to(self.device))
                        # loss = loss / 100
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                    del fvs, ents, batch_samples, inputs, outputs
                    torch.cuda.empty_cache()

                    # write to SummaryWriter
                    self.writer.add_scalar("loss", loss.item(), itr)
                    self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], itr)

                # write to SummaryWriter
                if self.train_bert:
                    self.bert_model.eval()
                self.mlp.eval()
                # import pdb; pdb.set_trace()

                print("train data evaluation...")
                tr_acc, tr_rec, _, tr_prec, tr_f1 = self.eval(self.train_dataset)
                print(f"acc: {tr_acc}, rec: {tr_rec}, prec: {tr_prec}, f1: {tr_f1}")
                self.writer.add_scalar("train/acc", tr_acc, epoch)
                self.writer.add_scalar("train/rec", tr_rec, epoch)
                self.writer.add_scalar("train/prec", tr_prec, epoch)
                self.writer.add_scalar("train/f1", tr_f1, epoch)
                # print("dev data evaluation...")
                # dev_acc, dev_rec, _, dev_prec, dev_f1 = self.eval(self.dev_dataset)
                # print(f"acc: {dev_acc}, rec: {dev_rec}, prec: {dev_prec}, f1: {dev_f1}")
                # self.writer.add_scalar("dev/acc", dev_acc, epoch)
                # self.writer.add_scalar("dev/rec", dev_rec, epoch)
                # self.writer.add_scalar("dev/prec", dev_prec, epoch)
                # self.writer.add_scalar("dev/f1", dev_f1, epoch)
                # self.writer.flush()
                print("dev1 data evaluation...")
                dev1_acc, dev1_rec, _, dev1_prec, dev1_f1 = self.eval(self.dev1_dataset)
                print(f"acc: {dev1_acc}, rec: {dev1_rec}, prec: {dev1_prec}, f1: {dev1_f1}")
                self.writer.add_scalar("dev1/acc", dev1_acc, epoch)
                self.writer.add_scalar("dev1/rec", dev1_rec, epoch)
                self.writer.add_scalar("dev1/prec", dev1_prec, epoch)
                self.writer.add_scalar("dev1/f1", dev1_f1, epoch)
                self.writer.flush()
                print("dev2 data evaluation...")
                dev2_acc, dev2_rec, _, dev2_prec, dev2_f1 = self.eval(self.dev2_dataset)
                print(f"acc: {dev2_acc}, rec: {dev2_rec}, prec: {dev2_prec}, f1: {dev2_f1}")
                self.writer.add_scalar("dev2/acc", dev2_acc, epoch)
                self.writer.add_scalar("dev2/rec", dev2_rec, epoch)
                self.writer.add_scalar("dev2/prec", dev2_prec, epoch)
                self.writer.add_scalar("dev2/f1", dev2_f1, epoch)
                self.writer.flush()
                if self.train_bert:
                    self.bert_model.train()
                self.mlp.train()

                if epoch % self.model_save_freq == 0:
                    curr_log_dir = self.log_dir / f"epoch_{epoch}"
                    if not curr_log_dir.exists():
                        curr_log_dir.mkdir()
                    if self.train_bert:
                        self.bert_model.save_pretrained(curr_log_dir)
                    torch.save(self.mlp.state_dict(), curr_log_dir / "mlp.model")

                # if best_dev_f1 <= dev_f1:
                #     best_dev_f1 = dev_f1
                #     best_dev_epoch = epoch
                #     if self.train_bert:
                #         best_dev_model = copy.deepcopy(self.bert_model)
                #     best_dev_mlp = copy.deepcopy(self.mlp.state_dict())
                if best_dev1_f1 <= dev1_f1:
                    best_dev1_f1 = dev1_f1
                    best_dev1_epoch = epoch
                    if self.train_bert:
                        best_dev1_model = copy.deepcopy(self.bert_model).cpu()
                    best_dev1_mlp = copy.deepcopy(self.mlp).cpu().state_dict()
                if best_dev2_f1 <= dev2_f1:
                    best_dev2_f1 = dev2_f1
                    best_dev2_epoch = epoch
                    if self.train_bert:
                        best_dev2_model = copy.deepcopy(self.bert_model).cpu()
                    best_dev2_mlp = copy.deepcopy(self.mlp).cpu().state_dict()

        except KeyboardInterrupt:
            # del fvs, ents, batch_samples, inputs, outputs
            # print(f"Best epoch was #{best_dev_epoch}!\nSave params...")
            # save_dev_dir = Path(self.log_dir) / "best"
            # if not save_dev_dir.exists():
            #     save_dev_dir.mkdir()
            # if self.train_bert:
            #     best_dev_model.save_pretrained(save_dev_dir)
            # torch.save(best_dev_mlp, save_dev_dir / "mlp.model")
            # print("Training was successfully finished!")
            print(f"Best epoch was dev1: #{best_dev1_epoch}, dev2: #{best_dev2_epoch}!\nSave params...")
            save_dev1_dir = Path(self.log_dir) / "dev1_best"
            if not save_dev1_dir.exists():
                save_dev1_dir.mkdir()
            save_dev2_dir = Path(self.log_dir) / "dev2_best"
            if not save_dev2_dir.exists():
                save_dev2_dir.mkdir()
            if self.train_bert:
                best_dev1_model.save_pretrained(save_dev1_dir)
                best_dev2_model.save_pretrained(save_dev2_dir)
            torch.save(best_dev1_mlp, save_dev1_dir / "mlp.model")
            torch.save(best_dev2_mlp, save_dev2_dir / "mlp.model")
            print("Training was successfully finished!")
            raise KeyboardInterrupt
        else:
            # print(f"Best epoch was #{best_dev_epoch}!\nSave params...")
            # save_dev_dir = Path(self.log_dir) / "best"
            # if not save_dev_dir.exists():
            #     save_dev_dir.mkdir()
            # if self.train_bert:
            #     best_dev_model.save_pretrained(save_dev_dir)
            # torch.save(best_dev_mlp, save_dev_dir / "mlp.model")
            # print("Training was successfully finished!")
            print(f"Best epoch was dev1: #{best_dev1_epoch}, dev2: #{best_dev2_epoch}!\nSave params...")
            save_dev1_dir = Path(self.log_dir) / "dev1_best"
            if not save_dev1_dir.exists():
                save_dev1_dir.mkdir()
            save_dev2_dir = Path(self.log_dir) / "dev2_best"
            if not save_dev2_dir.exists():
                save_dev2_dir.mkdir()
            if self.train_bert:
                best_dev1_model.save_pretrained(save_dev1_dir)
                best_dev2_model.save_pretrained(save_dev2_dir)
            torch.save(best_dev1_mlp, save_dev1_dir / "mlp.model")
            torch.save(best_dev2_mlp, save_dev2_dir / "mlp.model")
            print("Training was successfully finished!")
            sys.exit()


if __name__ == "__main__":
    config_path = "./configs/train.conf"
    Trainer = Trainer(config_path)
    Trainer.train()
