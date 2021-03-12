import argparse
from pathlib import Path
import spacy
from spacy.attrs import ORTH
from collections import Counter
import pandas as pd
from tqdm import tqdm
import json


def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("source_dir", type=str,
                        help="Directory path where ann files are stored")
    parser.add_argument("target_dir", type=str,
                        help="Directory path where jsonl files will be stored")
    parser.add_argument("dataset", type=str,
                        help="Dataset name")
    parser.add_argument("--main", action='store_true',
                        help="Specify this to distinguish between 'MAIN' and 'ELEMENT'. If it is not specified, 'MAIN' is treated as the same as 'ELEMENT'.")
    return parser.parse_args()


class ANN2DyGIEpp:
    def __init__(self, source_dir, target_dir, dataset, main):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True)
        self.dataset = dataset
        self.main_flag = main

        self.nlp = spacy.load("en_core_sci_sm")
        brackets = [r"\(", r"\)", r"\[", r"]", "{", "}"]
        equals = ["=", "＝", "≠", "≡", "≢", "~", "∼", "≑", "≒", "≓", "≃", "≈", "≔", "≕", "≝", "≜", "⋍", "≅", "˜"]
        ineqals = ["<", ">", "≤", "≥", "≦", "≧", "≪", "≫", "≲", "≳", "≶", "≷", "⋚", "⋛", "⋜", "⋝", "⩽", "⩾", "⪅", "⪆", "⪋", "⪌", "⪍", "⪎", "⪙", "⪚", "＜", "＞", "⋞", "⋟", "⪡", "⪢"]
        hyphens = ["-", "‐", "‑", "–", "—", "―", "−"]
        dashes = ["‒", "–", "—", "―", "⁓", "〜", "〰"]
        props = ["∝", "∼"]
        others = ["!", '"', "#", "$", "%", "&", "'", r"\^", r"\|", "@", r"\:", ";", r"\+", "±", "∓", "/", "_", r"\?", ",", "⊥", "", "‖", "→", "←", "↔", "⇄", "⇒", "⇔"]
        symbols = brackets + equals + ineqals + hyphens + dashes + props + others
        # add prefix pattern
        prefixes = list(self.nlp.Defaults.prefixes)
        prefixes.extend(symbols)
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        self.nlp.tokenizer.prefix_search = prefix_regex.search
        # add sufix pattern
        suffixes = list(self.nlp.Defaults.suffixes)
        suffixes.extend(symbols)
        suffixes.append(r"\.")
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        self.nlp.tokenizer.suffix_search = suffix_regex.search
        # add infix pattern
        infixes = list(self.nlp.Defaults.infixes)
        infixes.extend(symbols)
        infix_regex = spacy.util.compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_regex.finditer
        # add special case to tokenizer (BEFORE: "T c." -> "T" "c.", AFTER: "T c." -> "T" "c" ".")
        for l_alph in [chr(i) for i in range(97, 97+26)]:
            case = [{ORTH: f"{l_alph}"}, {ORTH: "."}]
            self.nlp.tokenizer.add_special_case(f"{l_alph}.", case)

        self.COUNTS = Counter()

    def read_txt(self, txt_p):
        with txt_p.open("r") as txt_f:
            text = txt_f.read()
        return text

    def read_ann(self, ann_p):
        entities, relations, events = [], [], []
        with ann_p.open("r") as ann_f:
            lines = ann_f.readlines()
        for l in lines:
            if l.startswith("T"):
                tid, label_start_end, mtn = l.strip().split("\t")
                label, start, end = [int(s) if i else s for i, s in enumerate(label_start_end.split())]
                if not self.main_flag:
                    # convert Main -> Element
                    if label == "Main":
                        label = "Element"
                entities.append({"id": tid, "label": label, "start": start, "end": end, "text": mtn})
            elif l.startswith("E"):
                curr_event = {}
                for i, e in enumerate(l.strip().split()):
                    if i == 0:
                        curr_event["id"] = e
                    elif i == 1:
                        _, trigger = e.split(":")
                        curr_event["trigger_id"] = trigger
                    else:
                        arg_name, arg = e.split(":")
                        curr_event[arg_name] = arg
                events.append(curr_event)
        for l in lines:
            if l.startswith("R"):
                rid, label, arg1, arg2 = l.strip().split()
                # ignore "coref" relation
                if label == "Coref":
                    continue
                arg1 = arg1.replace("Arg1:", "")
                arg2 = arg2.replace("Arg2:", "")
                if arg1.startswith("E"):
                    for evt in events:
                        if evt["id"] == arg1:
                            arg1 = evt["trigger_id"]
                            break
                if arg2.startswith("E"):
                    for evt in events:
                        if evt["id"] == arg2:
                            arg2 = evt["trigger_id"]
                            break
                relations.append({"id": rid, "label": label, "arg1": arg1, "arg2": arg2})
        return entities, relations, events

    def get_entities_in_sent(self, sent, entities):
        start, end = sent.start_char, sent.end_char
        res = []
        for ent in entities:
            if start <= ent["start"] and ent["end"] <= end:
                res.append(ent)
        return res

    def align_one(self, sent, ent):
        # Don't distinguish b/w genes that can and can't be looked up in database.
        start_tok = None
        end_tok = None
        for tok in sent:
            if tok.idx == ent["start"]:
                start_tok = tok
            if tok.idx + len(tok) == ent["end"]:
                end_tok = tok

        if start_tok is None or end_tok is None:
            return None
        else:
            expected = sent[start_tok.i - sent.start:end_tok.i - sent.start + 1]
            if expected.text != ent["text"]:
                raise Exception("Entity mismatch")
            return (start_tok.i, end_tok.i, ent["label"])

    def align_entities(self, sent, entities_sent):
        aligned_entities = {}
        missed_entities = {}
        for ent in entities_sent:
            aligned = self.align_one(sent, ent)
            if aligned is not None:
                aligned_entities[ent["id"]] = aligned
            else:
                # missed_entities[ent["id"]] = None
                missed_entities[ent["id"]] = ent
        return aligned_entities, missed_entities

    def format_relations(self, relations):
        # Convert to dict.
        res = {}
        for rel in relations:
            key = (rel["arg1"], rel["arg2"])
            res[key] = rel["label"]
        return res

    def format_events(self, events):
        res = {}
        for evt in events:
            key = (evt["id"], evt["trigger_id"])
            res[key] = []
            for k, v in evt.items():
                if k not in ("id", "trigger_id"):
                    res[key].append((v, k))
        return res

    def get_relations_in_sent(self, aligned, relations):
        res = []
        keys = set()
        # Loop over the relations, and keep the ones relating entities in this sentences.
        for ents, label in relations.items():
            if ents[0] in aligned and ents[1] in aligned:
                keys.add(ents)
                ent1 = aligned[ents[0]]
                ent2 = aligned[ents[1]]
                to_append = ent1[:2] + ent2[:2] + (label,)
                res.append(to_append)
        return res, keys

    def get_events_in_sent(self, aligned, events):
        res = []
        keys = set()
        for key, args in events.items():
            _, trigger_id = key
            if trigger_id in aligned:
                keys.add(key)
                trigger = aligned[trigger_id]
                to_append = [(trigger[0],) + (trigger[-1],)]
                for arg_id, arg_name in args:
                    if arg_id in aligned:
                        arg = aligned[arg_id]
                        to_append.append(arg[:2] + (arg_name,))
                res.append(to_append)
        return res, keys

    def one_abstract(self, doc_key, text, ann):
        entities, relations, events = ann
        relations = self.format_relations(relations)
        events = self.format_events(events)
        doc = self.nlp(text)

        entities_seen = set()
        entities_alignment = set()
        entities_no_alignment = set()
        relations_found = set()
        events_found = set()

        dygiepp_format = {"doc_key": doc_key, "dataset": self.dataset, "sentences": [], "ner": [], "relations": [], "events": []}

        for sent in doc.sents:
            # Get the tokens.
            toks = [tok.text for tok in sent]

            # Align entities.
            entities_sent = self.get_entities_in_sent(sent, entities)
            aligned, missed = self.align_entities(sent, entities_sent)

            # Align relations.
            relations_sent, rel_keys_found = self.get_relations_in_sent(aligned, relations)

            # Align events.
            events_sent, eve_keys_found = self.get_events_in_sent(aligned, events)

            # Append to result list
            dygiepp_format["sentences"].append(toks)
            entities_to_scierc = [list(x) for x in aligned.values()]
            dygiepp_format["ner"].append(entities_to_scierc)
            dygiepp_format["relations"].append(relations_sent)
            dygiepp_format["events"].append(events_sent)

            # Keep track of which entities and relations we've found and which we haven't.
            entities_seen |= set([e["id"] for e in entities_sent])
            entities_alignment |= set(aligned.keys())
            entities_no_alignment |= set(missed.keys())
            relations_found |= rel_keys_found
            events_found |= eve_keys_found

        # Update counts.
        entities_missed = set([e["id"] for e in entities_sent]) - entities_seen
        relations_missed = set(relations.keys()) - relations_found
        events_missed = set(events.keys()) - events_found

        self.COUNTS["entities_correct"] += len(entities_alignment)
        self.COUNTS["entities_misaligned"] += len(entities_no_alignment)
        self.COUNTS["entities_missed"] += len(entities_missed)
        self.COUNTS["entities_total"] += len(entities)
        self.COUNTS["relations_found"] += len(relations_found)
        self.COUNTS["relations_missed"] += len(relations_missed)
        self.COUNTS['relations_total'] += len(relations)
        self.COUNTS["events_found"] += len(events_found)
        self.COUNTS["events_missed"] += len(events_missed)
        self.COUNTS['events_total'] += len(events)

        return dygiepp_format

    def one_fold(self, fold_dir):
        fold = fold_dir.name
        print(f"Processing fold {fold}.")
        ann_files = list(fold_dir.glob("*.ann"))
        res = []
        for ann_p in tqdm(ann_files, total=len(ann_files)):
            doc_key = ann_p.stem
            ann = self.read_ann(ann_p)
            text = self.read_txt(ann_p.with_suffix(".txt"))
            to_append = self.one_abstract(doc_key, text, ann)
            res.append(to_append)

        # Write to file.
        target_p = self.target_dir / f"{fold}.jsonl"
        with open(target_p, "w") as f_out:
            for line in res:
                print(json.dumps(line), file=f_out)

    def format(self):
        source_subdirs = [p for p in self.source_dir.iterdir() if p.is_dir()]
        for fold_dir in source_subdirs:
            self.one_fold(fold_dir)
        counts = pd.Series(self.COUNTS)
        print()
        print("Some entities were missed due to tokenization choices in SciSpacy. Here are the stats:")
        print(counts)


if __name__ == "__main__":
    args = get_args()
    df = ANN2DyGIEpp(**vars(args))
    df.format()
