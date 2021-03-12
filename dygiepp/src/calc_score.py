import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("pred_dir", type=str, help="")
    parser.add_argument("gold_dir", type=str, help="")
    parser.add_argument("save_dir", type=str, help="")
    parser.add_argument("--main", action='store_true', help="")
    return parser.parse_args()


class Score:
    def __init__(self, pred_dir, gold_dir, save_dir, main):
        self.pred_dir = Path(pred_dir)
        assert self.pred_dir.is_dir()
        self.gold_dir = Path(gold_dir)
        assert self.gold_dir.is_dir()
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.save_path = self.save_dir / (self.pred_dir.name + ".csv")
        self.main_flag = main
        self.ent_labels = set()
        self.rel_labels = set()
        self.trig_labels = set()
        self.arg_labels = set()
        self.COUNT = Counter()

    def _read_ann(self, ann_p):
        assert ann_p.is_file()
        with ann_p.open("r") as ann_f:
            lines = ann_f.readlines()
        entities, events = {}, {}
        for l in lines:
            if l.startswith("T"):
                tid, label_start_end, mtn = l.strip().split("\t")
                label, start, end = [int(s) if i else s for i, s in enumerate(label_start_end.split())]
                entities[tid] = {"label": label, "start": start, "end": end}
            elif l.startswith("E"):
                curr_event = {}
                for i, e in enumerate(l.strip().split()):
                    if i == 0:
                        eid = e
                    elif i == 1:
                        _, trigger = e.split(":")
                        curr_event["trigger"] = entities[trigger]
                    else:
                        arg_name, arg = e.split(":")
                        curr_event[arg_name] = entities[arg]
                events[eid] = curr_event
        relations = []
        for l in lines:
            if l.startswith("R"):
                _, label, arg1, arg2 = l.strip().split()
                arg1 = arg1.replace("Arg1:", "")
                arg2 = arg2.replace("Arg2:", "")
                if arg1.startswith("E"):
                    arg1 = events[arg1]["trigger"]
                else:
                    arg1 = entities[arg1]
                if arg2.startswith("E"):
                    arg2 = events[arg2]["trigger"]
                else:
                    arg2 = entities[arg2]
                relations.append({"label": label, "arg1": arg1, "arg2": arg2})
        return list(entities.values()), relations, events

    def _entity(self, pred_ents, gold_ents):
        for pred_ent in pred_ents:
            label = pred_ent["label"]
            if pred_ent in gold_ents:
                self.COUNT["ent_tp"] += 1
                self.COUNT[f"ent_{label}_tp"] += 1
            else:
                self.COUNT["ent_fp"] += 1
                self.COUNT[f"ent_{label}_fp"] += 1
        for gold_ent in gold_ents:
            label = gold_ent["label"]
            self.ent_labels.add(label)
            if gold_ent not in pred_ents:
                self.COUNT["ent_fn"] += 1
                self.COUNT[f"ent_{label}_fn"] += 1

    def _relation(self, pred_rels, gold_rels):
        # exact
        pred_rels_no_ent_label = []
        for pred_rel in pred_rels:
            label = pred_rel["label"]
            temp = {}
            temp["label"] = label
            temp["arg1"] = {"start": pred_rel["arg1"]["start"], "end": pred_rel["arg1"]["end"]}
            temp["arg2"] = {"start": pred_rel["arg2"]["start"], "end": pred_rel["arg2"]["end"]}
            pred_rels_no_ent_label.append(temp)
            if pred_rel in gold_rels:
                self.COUNT["rel_exact_tp"] += 1
                self.COUNT[f"rel_exact_{label}_tp"] += 1
            else:
                self.COUNT["rel_exact_fp"] += 1
                self.COUNT[f"rel_exact_{label}_fp"] += 1
        gold_rels_no_ent_label = []
        for gold_rel in gold_rels:
            label = gold_rel["label"]
            temp = {}
            temp["label"] = label
            temp["arg1"] = {"start": gold_rel["arg1"]["start"], "end": gold_rel["arg1"]["end"]}
            temp["arg2"] = {"start": gold_rel["arg2"]["start"], "end": gold_rel["arg2"]["end"]}
            gold_rels_no_ent_label.append(temp)
            self.rel_labels.add(label)
            if gold_rel not in pred_rels:
                self.COUNT["rel_exact_fn"] += 1
                self.COUNT[f"rel_exact_{label}_fn"] += 1
        # rough
        for pred_rel in pred_rels_no_ent_label:
            label = pred_rel["label"]
            if pred_rel in gold_rels_no_ent_label:
                self.COUNT["rel_rough_tp"] += 1
                self.COUNT[f"rel_rough_{label}_tp"] += 1
            else:
                self.COUNT["rel_rough_fp"] += 1
                self.COUNT[f"rel_rough_{label}_fp"] += 1
        for gold_rel in gold_rels_no_ent_label:
            label = gold_rel["label"]
            if gold_rel not in pred_rels_no_ent_label:
                self.COUNT["rel_rough_fn"] += 1
                self.COUNT[f"rel_rough_{label}_fn"] += 1

    def _event(self, pred_evts, gold_evts):
        # trigger
        pred_trigs = [item["trigger"] for item in pred_evts.values()]
        gold_trigs = [item["trigger"] for item in gold_evts.values()]
        for pred_trig in pred_trigs:
            label = pred_trig["label"]
            if pred_trig in gold_trigs:
                self.COUNT["trig_tp"] += 1
                self.COUNT[f"trig_{label}_tp"] += 1
            else:
                self.COUNT["trig_fp"] += 1
                self.COUNT[f"trig_{label}_fp"] += 1
        for gold_trig in gold_trigs:
            label = gold_trig["label"]
            self.trig_labels.add(label)
            if gold_trig not in pred_trigs:
                self.COUNT["trig_fn"] += 1
                self.COUNT[f"trig_{label}_fn"] += 1
        # make tirigger/argument pairs
        exact_pred_trig_arg_pairs = []
        rough_pred_trig_arg_pairs = []
        for pred_evt in pred_evts.values():
            trig = pred_evt["trigger"]
            rough_trig = {"start": trig["start"], "end": trig["end"]}
            for arg_name, arg in pred_evt.items():
                if arg_name != "trigger":
                    rough_arg = {"start": arg["start"], "end": arg["end"]}
                    exact_pred_trig_arg_pairs.append((arg_name, trig, arg))
                    rough_pred_trig_arg_pairs.append((arg_name, rough_trig, rough_arg))
        exact_gold_trig_arg_pairs = []
        rough_gold_trig_arg_pairs = []
        for gold_evt in gold_evts.values():
            trig = gold_evt["trigger"]
            rough_trig = {"start": trig["start"], "end": trig["end"]}
            for arg_name, arg in gold_evt.items():
                if arg_name != "trigger":
                    self.arg_labels.add(arg_name)
                    rough_arg = {"start": arg["start"], "end": arg["end"]}
                    exact_gold_trig_arg_pairs.append((arg_name, trig, arg))
                    rough_gold_trig_arg_pairs.append((arg_name, rough_trig, rough_arg))
        # argument (exact)
        for pred_pair in exact_pred_trig_arg_pairs:
            arg_name = pred_pair[0]
            if pred_pair in exact_gold_trig_arg_pairs:
                self.COUNT["arg_exact_tp"] += 1
                self.COUNT[f"arg_exact_{arg_name}_tp"] += 1
            else:
                self.COUNT["arg_exact_fp"] += 1
                self.COUNT[f"arg_exact_{arg_name}_fp"] += 1
        for gold_pair in exact_gold_trig_arg_pairs:
            arg_name = gold_pair[0]
            if gold_pair not in exact_pred_trig_arg_pairs:
                self.COUNT["arg_exact_fn"] += 1
                self.COUNT[f"arg_exact_{arg_name}_fn"] += 1
        # argument (rough)
        for pred_pair in rough_pred_trig_arg_pairs:
            arg_name = pred_pair[0]
            if pred_pair in rough_gold_trig_arg_pairs:
                self.COUNT["arg_rough_tp"] += 1
                self.COUNT[f"arg_rough_{arg_name}_tp"] += 1
            else:
                self.COUNT["arg_rough_fp"] += 1
                self.COUNT[f"arg_rough_{arg_name}_fp"] += 1
        for gold_pair in rough_gold_trig_arg_pairs:
            arg_name = gold_pair[0]
            if gold_pair not in rough_pred_trig_arg_pairs:
                self.COUNT["arg_rough_fn"] += 1
                self.COUNT[f"arg_rough_{arg_name}_fn"] += 1
        # total
        for pred_evt in pred_evts.values():
            trig = pred_evt["trigger"]["label"]
            if pred_evt in gold_evts.values():
                self.COUNT["evt_tp"] += 1
                self.COUNT[f"evt_{trig}_tp"] += 1
            else:
                self.COUNT["evt_fp"] += 1
                self.COUNT[f"evt_{trig}_fp"] += 1
        for gold_evt in gold_evts.values():
            trig = gold_evt["trigger"]["label"]
            if gold_evt not in pred_evts.values():
                self.COUNT["evt_fn"] += 1
                self.COUNT[f"evt_{trig}_fn"] += 1

    def _calc_metrics(self, counts):
        tp, fp, fn = counts
        try:
            rec = tp / (tp + fn)
        except ZeroDivisionError:
            rec = 0.
        try:
            prec = tp / (tp + fp)
        except ZeroDivisionError:
            prec = 0.
        try:
            fscore = 2 * rec * prec / (rec + prec)
        except ZeroDivisionError:
            fscore = 0.
        return rec, prec, fscore

    def calc(self):
        for pred_p in self.pred_dir.glob("*.ann"):
            gold_p = self.gold_dir / pred_p.name
            pred_ents, pred_rels, pred_evts = self._read_ann(pred_p)
            gold_ents, gold_rels, gold_evts = self._read_ann(gold_p)

            if not self.main_flag:
                for pred_ent in pred_ents:
                    if pred_ent["label"] == "Main":
                        pred_ent["label"] = "Element"
                for gold_ent in gold_ents:
                    if gold_ent["label"] == "Main":
                        gold_ent["label"] = "Element"

            self._entity(pred_ents, gold_ents)
            self._relation(pred_rels, gold_rels)
            self._event(pred_evts, gold_evts)

        csv_lines = ["Category,Precision,Recall,F-score"]
        print("### Result ###")
        print(f"gold: {self.gold_dir}")
        print(f"pred: {self.pred_dir}")
        print()
        print("## Entity")
        csv_lines.append("## Entity ##")
        ent_res = self._calc_metrics((self.COUNT["ent_tp"], self.COUNT["ent_fp"], self.COUNT["ent_fn"]))
        print(f"rec: {ent_res[0]:.4f}\tprec: {ent_res[1]:.4f}\tfscore: {ent_res[2]:.4f}")
        for label in self.ent_labels:
            label_res = self._calc_metrics((self.COUNT[f"ent_{label}_tp"], self.COUNT[f"ent_{label}_fp"], self.COUNT[f"ent_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")
        print()
        print("## Relation")
        csv_lines.append("## Relation ##")
        print("# exact")
        csv_lines.append("# exact #")
        rel_exact_res = self._calc_metrics((self.COUNT["rel_exact_tp"], self.COUNT["rel_exact_fp"], self.COUNT["rel_exact_fn"]))
        print(f"rec: {rel_exact_res[0]:.4f}\tprec: {rel_exact_res[1]:.4f}\tfscore: {rel_exact_res[2]:.4f}")
        for label in self.rel_labels:
            label_res = self._calc_metrics((self.COUNT[f"rel_exact_{label}_tp"], self.COUNT[f"rel_exact_{label}_fp"], self.COUNT[f"rel_exact_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")
        print("# rough")
        csv_lines.append("# rough #")
        rel_rough_res = self._calc_metrics((self.COUNT["rel_rough_tp"], self.COUNT["rel_rough_fp"], self.COUNT["rel_rough_fn"]))
        print(f"rec: {rel_rough_res[0]:.4f}\tprec: {rel_rough_res[1]:.4f}\tfscore: {rel_rough_res[2]:.4f}")
        for label in self.rel_labels:
            label_res = self._calc_metrics((self.COUNT[f"rel_rough_{label}_tp"], self.COUNT[f"rel_rough_{label}_fp"], self.COUNT[f"rel_rough_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")
        print()
        print("## Event")
        csv_lines.append("## Event ##")
        print("# trigger")
        csv_lines.append("# tigger #")
        trig_res = self._calc_metrics((self.COUNT["trig_tp"], self.COUNT["trig_fp"], self.COUNT["trig_fn"]))
        print(f"rec: {trig_res[0]:.4f}\tprec: {trig_res[1]:.4f}\tfscore: {trig_res[2]:.4f}")
        for label in self.trig_labels:
            label_res = self._calc_metrics((self.COUNT[f"trig_{label}_tp"], self.COUNT[f"trig_{label}_fp"], self.COUNT[f"trig_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")
        print("# arg_exact")
        csv_lines.append("# arg_exact #")
        arg_exact_res = self._calc_metrics((self.COUNT["arg_exact_tp"], self.COUNT["arg_exact_fp"], self.COUNT["arg_exact_fn"]))
        print(f"rec: {arg_exact_res[0]:.4f}\tprec: {arg_exact_res[1]:.4f}\tfscore: {arg_exact_res[2]:.4f}")
        for label in self.arg_labels:
            label_res = self._calc_metrics((self.COUNT[f"arg_exact_{label}_tp"], self.COUNT[f"arg_exact_{label}_fp"], self.COUNT[f"arg_exact_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")
        print("# arg_rough")
        csv_lines.append("# arg_rough #")
        arg_rough_res = self._calc_metrics((self.COUNT["arg_rough_tp"], self.COUNT["arg_rough_fp"], self.COUNT["arg_rough_fn"]))
        print(f"rec: {arg_rough_res[0]:.4f}\tprec: {arg_rough_res[1]:.4f}\tfscore: {arg_rough_res[2]:.4f}")
        for label in self.arg_labels:
            label_res = self._calc_metrics((self.COUNT[f"arg_rough_{label}_tp"], self.COUNT[f"arg_rough_{label}_fp"], self.COUNT[f"arg_rough_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")
        print("# total")
        csv_lines.append("# total #")
        evt_res = self._calc_metrics((self.COUNT["evt_tp"], self.COUNT["evt_fp"], self.COUNT["evt_fn"]))
        print(f"rec: {evt_res[0]:.4f}\tprec: {evt_res[1]:.4f}\tfscore: {evt_res[2]:.4f}")
        for label in self.trig_labels:
            label_res = self._calc_metrics((self.COUNT[f"evt_{label}_tp"], self.COUNT[f"evt_{label}_fp"], self.COUNT[f"evt_{label}_fn"]))
            csv_lines.append(f"{label},{label_res[1]:.4f},{label_res[0]:.4f},{label_res[2]:.4f}")
            print(f"@{label}::\trec: {label_res[0]:.4f}\tprec: {label_res[1]:.4f}\tfscore: {label_res[2]:.4f}")

        with self.save_path.open("w") as f:
            f.write("\n".join(csv_lines))


if __name__ == "__main__":
    args = get_args()
    score = Score(**vars(args))
    score.calc()
