import argparse
import copy
import itertools
import json
import yaml
import re
from pathlib import Path
from tqdm import tqdm
from ann_parser import ANNParser


def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("data_name", type=str, help="")
    parser.add_argument("data_dir", type=str, help="")
    parser.add_argument("save_dir", type=str, help="")
    return parser.parse_args()


class ANNtoJSON():
    def __init__(self, data_name, data_dir, save_dir):
        self.data_name = data_name
        self.data_dir = Path(data_dir)
        assert self.data_dir.is_dir()
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        with open("./elements.yml") as yml:
            self.elements = yaml.load(yml)

    def read_txt(self, txt_p):
        with txt_p.open() as txt_f:
            text = txt_f.read()
        return text

    def read_ann(self, ann_p):
        entities, relations, events = {}, [], {}
        with ann_p.open("r") as ann_f:
            lines = ann_f.readlines()
        for l in lines:
            if l.startswith("T"):
                tid, label_start_end, mtn = l.strip().split("\t")
                label, start, end = [int(s) if i else s for i, s in enumerate(label_start_end.split())]
                entities[tid] = {"label": label, "start": start, "end": end, "text": mtn}
            elif l.startswith("R"):
                rid, label, arg1, arg2 = l.strip().split()
                arg1 = arg1.replace("Arg1:", "")
                arg2 = arg2.replace("Arg2:", "")
                relations.append({"label": label, "arg1": arg1, "arg2": arg2})
            elif l.startswith("E"):
                curr_event = {}
                for i, e in enumerate(l.strip().split()):
                    if i == 0:
                        eid = e
                    elif i == 1:
                        _, trigger = e.split(":")
                        curr_event["trigger"] = trigger
                    else:
                        arg_name, arg = e.split(":")
                        curr_event[arg_name] = arg
                events[eid] = curr_event
        return entities, relations, events

    def parse_composition(self, comp):
        comp = comp.replace(" ", "")
        elems_2 = [e for e in self.elements.keys() if len(e) == 2]
        upper_alph = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        elem_cands = elems_2 + upper_alph
        elems_reg = "(" + "|".join(elem_cands) + ")"
        comp_list = []
        while True:
            matched = re.match(elems_reg, comp)
            if matched:
                comp_list.append(comp[:matched.end()])
                comp = comp[matched.end():]
            else:
                searched = re.search(elems_reg, comp)
                if searched:
                    comp_list.append(comp[:searched.start()])
                    comp = comp[searched.start():]
                else:
                    if comp:
                        comp_list.append(comp)
                    break
        # ref: http://www13.plala.or.jp/bigdata/yokobou.html
        minus_syms = ["-", "－", "﹣", "−", "‐", "⁃", "‑", "‒", "–", "—", "﹘", "―", "⎯", "⏤", "˗"]
        var_syms = []
        for c in comp_list:
            # check dopant
            if re.match(r"^[a-zδ]$", c):
                var_syms.append(c)
            # check site
            for m in minus_syms:
                if re.match(rf"(\d\.)?\d+\s*{m}.*", c):
                    var_cand = re.split(rf"(\d\.)?\d+\s*{m}", c)[-1]
                    var_syms.extend(re.findall(r"[a-zδ]", var_cand))
                    break
        var_syms = set(var_syms)

        doping = {v: {"Dopant": [], "Site": []} for v in var_syms}
        for v in var_syms:
            for i, c in enumerate(comp_list):
                if re.match(rf"(\d\.)?\d*\s*{v}", c):
                    doping[v]["Dopant"].append(comp_list[i-1])
                    continue
                for m in minus_syms:
                    if re.match(rf"(\d\.)?\d+\s*{m}.*{v}.*", c):
                        doping[v]["Site"].append(comp_list[i-1])
                        break

        comp_elems = {}
        comp_list_ = comp_list + ["[END]"]
        for i, c in enumerate(comp_list_[:-1]):
            if c in elem_cands:
                if comp_list_[i+1] in elem_cands or comp_list_[i+1] == "[END]":
                    comp_elems[c] = 1.
                else:
                    try:
                        f = float(comp_list_[i+1])
                    except ValueError:
                        pass
                    else:
                        comp_elems[c] = f
                        continue

                    for v in var_syms:
                        if re.match(rf"(\d\.)?\d+\s*{v}", comp_list_[i+1]):
                            comp_elems[c] = 0.
                            break
                        if re.match(rf"(\d\.)?\d+.*{v}.*", comp_list_[i+1]):
                            comp_elems[c] = float(re.match(r"(\d\.)?\d+", comp_list_[i+1]).group())
                            break
                    else:
                        comp_elems[c] = 0.

        return doping, comp_elems

    def get_sentence_spans(self, text):
        sentence_spans = []
        start = 0
        for matched in re.finditer(r"\n", text):
            end, next_start = matched.span()
            sentence_spans.append((start, end))
            start = next_start
        else:
            sentence_spans.append((start, len(text)))
        return sentence_spans

    def get_sentence_id(self, sentence_spans, start, end):
        for i, span in enumerate(sentence_spans):
            s_start, s_end = span
            if s_start <= start and end < s_end:
                return i
        else:
            raise IndexError("start/end index out of range!")

    def get_main_info(self, main_ents, sentence_spans):
        main_info = []
        for s_start, s_end in sentence_spans:
            sent_main_info = []
            for ent in main_ents:
                _, e_start, e_end, _ = ent
                if s_start <= ent["start"] and ent["end"] < s_end:
                    sent_main_info.append(ent)
            main_info.append(sent_main_info)
        return main_info

    def slot_fliing(self):
        rel_res_slots = {"dataset": self.data_name, "docs": {}}
        main_res_slots = {"dataset": self.data_name, "docs": {}}
        all_res_slots = {"dataset": self.data_name, "docs": {}}
        ann_paths = sorted(self.data_dir.glob("*.ann"))
        for ann_p in tqdm(ann_paths, total=len(ann_paths)):
            fname = ann_p.name
            entities, relations, events = self.read_ann(ann_p)
            txt_p = ann_p.with_suffix(".txt")
            text = self.read_txt(txt_p)
            sent_spans = self.get_sentence_spans(text)

            curr_slots = []
            # Tc
            for rel in relations:
                if rel["label"] == "Equivalent":
                    arg1_id, arg2_id = rel["arg1"], rel["arg2"]
                    if arg1_id.startswith("T") and arg2_id.startswith("T"):
                        arg1, arg2 = entities[arg1_id], entities[arg2_id]
                        # when Tc slot can be filled
                        if arg1["label"] == "SC" and arg2["label"] == "Value":
                            sent_id = self.get_sentence_id(sent_spans, arg2["start"], arg2["end"])
                            slot = {"Element": [], "Tc": arg2, "Dopant": None, "Site": None, "sent_id": sent_id}
                            # fill "Element" slot
                            for rel_ in relations:
                                if rel_["label"] == "Target":
                                    if rel_["arg1"] == arg2_id and rel_["arg2"].startswith("T"):
                                        target = entities[rel_["arg2"]]
                                        if target["label"] in ("Element", "Main"):
                                            slot["Element"].append(target)
                            # in case of "Doping" is conditioned by "Tc"
                            for rel_ in relations:
                                if rel_["label"] == "Condition" and rel_["arg1"] == arg2_id and rel_["arg2"].startswith("E"):
                                    cond_arg = events[rel_["arg2"]]
                                    if entities[cond_arg["trigger"]]["label"] == "Doping":
                                        for arg_l, arg in cond_arg.items():
                                            if "Dopant" in arg_l:
                                                slot["Dopant"] = entities[arg]
                                            elif "Site" in arg_l:
                                                slot["Site"] = entities[arg]
                                        # fill "Element" when its slot is empty
                                        if slot["Element"] == []:
                                            for rel__ in relations:
                                                if rel__["label"] == "Target":
                                                    if rel__["arg1"] == rel_["arg2"] and rel__["arg2"].startswith("T"):
                                                        target = entities[rel__["arg2"]]
                                                        if target["label"] in ("Element", "Main"):
                                                            slot["Element"].append(target)
                            curr_slots.append(slot)

            # Doping
            for eid, evt in events.items():
                trigger = entities[evt["trigger"]]
                if trigger["label"] == "Doping":
                    sent_id = self.get_sentence_id(sent_spans, trigger["start"], trigger["end"])
                    slot = {"Element": [], "Tc": None, "Dopant": None, "Site": None, "sent_id": sent_id}
                    # fill "Dopant"/"Site" slot
                    try:
                        for arg_l, arg in evt.items():
                            if "Dopant" in arg_l:
                                slot["Dopant"] = entities[arg]
                            elif "Site" in arg_l:
                                slot["Site"] = entities[arg]
                    except KeyError:
                        continue
                    if slot["Dopant"] or slot["Site"]:
                        for rel in relations:
                            if rel["label"] == "Target":
                                if rel["arg1"] == eid and rel["arg2"].startswith("T"):
                                    target = entities[rel["arg2"]]
                                    if target["label"] in ("Element", "Main"):
                                        slot["Element"].append(target)
                        curr_slots.append(slot)

            rel_slots = []
            main_slots = []
            for curr_s in curr_slots:
                if curr_s["Element"]:
                    rel_slots.append(curr_s)
                else:
                    main_slots.append(curr_s)

            # Linking Tc/Doping information to the main materials
            main_ents = [ent for _, ent in entities.items() if ent["label"] == "Main"]

            if main_slots:
                if main_ents:
                    temp_slots = copy.copy(main_slots)
                    main_info = self.get_main_info(main_ents, sent_spans)
                    for i, slot in enumerate(temp_slots):
                        curr_sent_id = slot["sent_id"]
                        main_resolved = False
                        prev_mains = list(reversed(main_info[:curr_sent_id + 1]))
                        folw_mains = list(main_info[curr_sent_id:])
                        for p_mains in prev_mains:
                            if p_mains:
                                has_prev_mains = True
                                break
                        else:
                            has_prev_mains = False
                        # previous mains
                        if has_prev_mains:
                            for main_cands in prev_mains:
                                if main_cands == []:
                                    continue
                                else:
                                    if slot["Dopant"] and slot["Site"]:
                                        for main_cand in main_cands:
                                            doping, _ = self.parse_composition(main_cand["text"])
                                            for var, dpt_site in doping.items():
                                                if slot["Dopant"]["text"] in dpt_site["Dopant"] \
                                                    and slot["Site"]["text"] in dpt_site["Site"]:
                                                    main_slots[i]["Element"].append(main_cand)
                                                    main_resolved = True
                                        if main_resolved:
                                            break
                                    elif slot["Dopant"]:
                                        for main_cand in main_cands:
                                            doping, _ = self.parse_composition(main_cand["text"])
                                            for var, dpt_site in doping.items():
                                                if slot["Dopant"]["text"] in dpt_site["Dopant"]:
                                                    main_slots[i]["Element"].append(main_cand)
                                                    main_resolved = True
                                        if main_resolved:
                                            break
                                    elif slot["Site"]:
                                        for main_cand in main_cands:
                                            doping, _ = self.parse_composition(main_cand["text"])
                                            for var, dpt_site in doping.items():
                                                if slot["Site"]["text"] in dpt_site["Site"]:
                                                    main_slots[i]["Element"].append(main_cand)
                                                    main_resolved = True
                                        if main_resolved:
                                            break
                                    else:
                                        main_slots[i]["Element"] = main_cands
                                        break
                            else:
                                for main_cands in reversed(main_info[:curr_sent_id + 1]):
                                    if main_cands != []:
                                        main_slots[i]["Element"] = main_cands
                                        break
                        # following mains
                        else:
                            for main_cands in folw_mains:
                                if main_cands == []:
                                    continue
                                else:
                                    if slot["Dopant"] and slot["Site"]:
                                        for main_cand in main_cands:
                                            doping, _ = self.parse_composition(main_cand["text"])
                                            for var, dpt_site in doping.items():
                                                if slot["Dopant"]["text"] in dpt_site["Dopant"] \
                                                    and slot["Site"]["text"] in dpt_site["Site"]:
                                                    main_slots[i]["Element"].append(main_cand)
                                                    main_resolved = True
                                        if main_resolved:
                                            break
                                    elif slot["Dopant"]:
                                        for main_cand in main_cands:
                                            doping, _ = self.parse_composition(main_cand["text"])
                                            for var, dpt_site in doping.items():
                                                if slot["Dopant"]["text"] in dpt_site["Dopant"]:
                                                    main_slots[i]["Element"].append(main_cand)
                                                    main_resolved = True
                                        if main_resolved:
                                            break
                                    elif slot["Site"]:
                                        for main_cand in main_cands:
                                            doping, _ = self.parse_composition(main_cand["text"])
                                            for var, dpt_site in doping.items():
                                                if slot["Site"]["text"] in dpt_site["Site"]:
                                                    main_slots[i]["Element"].append(main_cand)
                                                    main_resolved = True
                                        if main_resolved:
                                            break
                                    else:
                                        main_slots[i]["Element"] = main_cands
                                        break
                            else:
                                for main_cands in main_info[curr_sent_id:]:
                                    if main_cands != []:
                                        main_slots[i]["Element"] = main_cands
                                        break

            rel_res_slots["docs"][fname] = []
            if rel_slots:
                for slot in rel_slots:
                    if slot["Element"]:
                        for elem in slot["Element"]:
                            res_slot = {"Element": None,
                                        "Tc": None,
                                        "Dopant": None,
                                        "Site": None}
                            t, s, e = elem["text"], elem["start"], elem["end"]
                            res_slot["Element"] = {"text": t, "start": s, "end": e}
                            if slot["Tc"]:
                                t, s, e = slot["Tc"]["text"], slot["Tc"]["start"], slot["Tc"]["end"]
                                res_slot["Tc"] = {"text": t, "start": s, "end": e}
                            if slot["Dopant"]:
                                t, s, e = slot["Dopant"]["text"], slot["Dopant"]["start"], slot["Dopant"]["end"]
                                res_slot["Dopant"] = {"text": t, "start": s, "end": e}
                            if slot["Site"]:
                                t, s, e = slot["Site"]["text"], slot["Site"]["start"], slot["Site"]["end"]
                                res_slot["Site"] = {"text": t, "start": s, "end": e}
                            rel_res_slots["docs"][fname].append(res_slot)
            main_res_slots["docs"][fname] = []
            if main_slots:
                for slot in main_slots:
                    if slot["Element"]:
                        for elem in slot["Element"]:
                            res_slot = {"Element": None,
                                        "Tc": None,
                                        "Dopant": None,
                                        "Site": None}
                            t, s, e = elem["text"], elem["start"], elem["end"]
                            res_slot["Element"] = {"text": t, "start": s, "end": e}
                            if slot["Tc"]:
                                t, s, e = slot["Tc"]["text"], slot["Tc"]["start"], slot["Tc"]["end"]
                                res_slot["Tc"] = {"text": t, "start": s, "end": e}
                            if slot["Dopant"]:
                                t, s, e = slot["Dopant"]["text"], slot["Dopant"]["start"], slot["Dopant"]["end"]
                                res_slot["Dopant"] = {"text": t, "start": s, "end": e}
                            if slot["Site"]:
                                t, s, e = slot["Site"]["text"], slot["Site"]["start"], slot["Site"]["end"]
                                res_slot["Site"] = {"text": t, "start": s, "end": e}
                            main_res_slots["docs"][fname].append(res_slot)

            # integrate relation/main slots
            all_doc_slots = []
            for slot in copy.deepcopy(rel_res_slots["docs"][fname] + main_res_slots["docs"][fname]):
                doping, comp_elems = self.parse_composition(slot["Element"]["text"])
                slot["Element"]["comp_elems"] = comp_elems
                all_doc_slots.append(slot)
            # dopant/siteを持つ全てのElementエンティティ情報を新たに追加
            for _, ent in entities.items():
                if ent["label"] in ("Element", "Main"):
                    doping, comp_elems = self.parse_composition(ent["text"])
                    if doping:
                        for var, dpt_site in doping.items():
                            dpts, sites = dpt_site["Dopant"], dpt_site["Site"]
                            if dpts and sites:
                                for dpt, site in itertools.product(dpts, sites):
                                    slot = {"Element": {"text": ent["text"], "start": ent["start"], "end": ent["end"], "comp_elems": comp_elems},
                                            "Tc": None,
                                            "Dopant": {"text": dpt, "start": None, "end": None},
                                            "Site": {"text": site, "start": None, "end": None}}
                                    all_doc_slots.append(slot)
                            elif dpts:
                                for dpt in dpts:
                                    slot = {"Element": {"text": ent["text"], "start": ent["start"], "end": ent["end"], "comp_elems": comp_elems},
                                            "Tc": None,
                                            "Dopant": {"text": dpt, "start": None, "end": None},
                                            "Site": None}
                                    all_doc_slots.append(slot)
                            elif sites:
                                for site in sites:
                                    slot = {"Element": {"text": ent["text"], "start": ent["start"], "end": ent["end"], "comp_elems": comp_elems},
                                            "Tc": None,
                                            "Dopant": None,
                                            "Site": {"text": site, "start": None, "end": None}}
                                    all_doc_slots.append(slot)
            all_res_slots["docs"][fname] = all_doc_slots
        return rel_res_slots, main_res_slots, all_res_slots

    def save_json(self, slots, file_name):
        save_path = self.save_dir / file_name
        with open(save_path, 'w') as f:
            json.dump(slots, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    ann2json = ANNtoJSON(**vars(args))

    rel_slots, main_slots, all_slots = ann2json.slot_fliing()
    ann2json.save_json(rel_slots, "relation.json")
    ann2json.save_json(main_slots, "main.json")
    ann2json.save_json(all_slots, "all.json")

    print("### number of slots ###")
    rel_c = 0
    for fname, s in rel_slots["docs"].items():
        rel_c += len(s)
    print("relation:", rel_c)
    main_c = 0
    for fname, s in main_slots["docs"].items():
        main_c += len(s)
    print("main:", main_c)
    all_c = 0
    for fname, s in all_slots["docs"].items():
        all_c += len(s)
    print("all:", all_c)
