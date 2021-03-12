import argparse
from pathlib import Path
from tqdm import tqdm
import json
import itertools


def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("source_path", type=str,
                        help="Path to the jsonl file to be converted")
    return parser.parse_args()


class DyGIEpp2ANN:
    def __init__(self, source_path):
        self.source_path = Path(source_path)
        assert self.source_path.is_file()

    def read_txt(self, txt_p):
        with txt_p.open("r") as txt_f:
            text = txt_f.read()
        return text

    def read_jsonl(self, jsonl_p):
        with jsonl_p.open("r") as jsonl_f:
            json_lines = jsonl_f.readlines()
        res = []
        for l in json_lines:
            res.append(json.loads(l))
        return res

    def tokid2index(self, text, sents):
        res = []
        tok_start, tok_end = 0, 0
        for sent in sents:
            sent_idxes = []
            for tok in sent:
                tok_start = tok_end + text[tok_end:].find(tok)
                tok_end = tok_start + len(tok)
                sent_idxes.append((tok_start, tok_end))
            res.append(sent_idxes)
        return res

    def ner(self, text, sents_ner, sents_idxes):
        tid = 1
        prev_ntoks = 0
        res = {}
        for i, sent_ner in enumerate(sents_ner):
            for ner in sent_ner:
                tok_start, tok_end, label = int(ner[0]), int(ner[1]), ner[2]
                ne_start = sents_idxes[i][tok_start-prev_ntoks][0]
                ne_end = sents_idxes[i][tok_end-prev_ntoks][1]
                mention = text[ne_start: ne_end]
                res[(tok_start, tok_end)] = (f"T{tid}", label, ne_start, ne_end, mention)
                tid += 1
            prev_ntoks += len(sents_idxes[i])
        return res

    def relation(self, sents_rel, sents_ner_dict, res_evt):
        rid = 1
        res = []
        for i, sent_rel in enumerate(sents_rel):
            for rel in sent_rel:
                arg1_start, arg1_end = int(rel[0]), int(rel[1])
                arg2_start, arg2_end = int(rel[2]), int(rel[3])
                label = rel[4]
                try:
                    arg1_tid = sents_ner_dict[(arg1_start, arg1_end)][0]
                    arg2_tid = sents_ner_dict[(arg2_start, arg2_end)][0]
                except KeyError:
                    continue
                else:
                    inc_event = False
                    for evt in res_evt:
                        if arg1_tid == evt[1][1]:
                            res.append((f"R{rid}", label, evt[0], arg2_tid))
                            inc_event = True
                            rid += 1
                        if arg2_tid == evt[1][1]:
                            res.append((f"R{rid}", label, arg1_tid, evt[0]))
                            inc_event = True
                            rid += 1
                    if not inc_event:
                        res.append((f"R{rid}", label, arg1_tid, arg2_tid))
                        rid += 1
        return res

    def event(self, sents_evt, sents_ner_dict):
        eid = 1
        res = []
        for i, sent_evt in enumerate(sents_evt):
            for evt in sent_evt:
                trigger, trigger_label = int(evt[0][0]), evt[0][1]
                try:
                    trigger_tid = sents_ner_dict[(trigger, trigger)][0]
                except KeyError:
                    continue
                args_dict = {}
                for arg in evt[1:]:
                    arg_start, arg_end, label = int(arg[0]), int(arg[1]), arg[2]
                    try:
                        arg_tid = sents_ner_dict[(arg_start, arg_end)][0]
                    except KeyError:
                        continue
                    else:
                        # temp_res.append((label, arg_tid))
                        if label not in args_dict:
                            args_dict[label] = [arg_tid]
                        else:
                            args_dict[label].append(arg_tid)
                args_product = [x for x in itertools.product(*args_dict.values())]
                product_list = [dict(zip(args_dict.keys(), r)) for r in args_product]
                for product_dict in product_list:
                    trigger_res = [f"E{eid}", (trigger_label, trigger_tid)]
                    args_res = [(label, arg_tid) for label, arg_tid in product_dict.items()]
                    res.append(trigger_res + args_res)
                    eid += 1
        return res

    def save_file(self, save_p, res_ner, res_rel, res_evt):
        ner_lines = [f"{l[0]}\t{l[1]} {l[2]} {l[3]}\t{l[4]}" for l in res_ner]
        rel_lines = [f"{l[0]}\t{l[1]} Arg1:{l[2]} Arg2:{l[3]}" for l in res_rel]
        evt_lines = []
        for l in res_evt:
            t = f"{l[0]}\t"
            temp = []
            for s in l[1:]:
                temp.append(f"{s[0]}:{s[1]}")
            evt_lines.append(t+" ".join(temp))
        with save_p.open("w") as save_f:
            save_f.write("\n".join(ner_lines + rel_lines + evt_lines))

    def format(self):
        target_dir = self.source_path.parent / self.source_path.stem
        assert target_dir.exists()
        # if not target_dir.exists():
        #     target_dir.mkdir(parents=True)
        doc_res = self.read_jsonl(self.source_path)
        for res in doc_res:
            doc_key = res["doc_key"]
            # print(doc_key)
            txt_p = target_dir / f"{doc_key}.txt"
            assert txt_p.exists()
            text = self.read_txt(txt_p)
            sents_idxes = self.tokid2index(text, res["sentences"])
            try:
                sents_ner_dict = self.ner(text, res["predicted_ner"], sents_idxes)
            except KeyError:
                continue
            res_ner = [v for v in sents_ner_dict.values()]
            res_evt = self.event(res["predicted_events"], sents_ner_dict)
            # import pdb; pdb.set_trace()
            res_rel = self.relation(res["predicted_relations"], sents_ner_dict, res_evt)
            save_p = target_dir / f"{doc_key}.ann"
            self.save_file(save_p, res_ner, res_rel, res_evt)


if __name__ == "__main__":
    args = get_args()
    df = DyGIEpp2ANN(**vars(args))
    df.format()
