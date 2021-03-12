from torchtext import data, datasets


def create_dataset(data_path):
    TEXT = data.Field()
    START = data.Field()
    END = data.Field()
    LABEL = data.Field()
    dataset = datasets.SequenceTaggingDataset(path=data_path,
                                              fields=[('label', LABEL), ('start', START), ('end', END), ('text', TEXT)])
    LABEL.build_vocab(dataset)
    label_list = list(LABEL.vocab.freqs) + ["X"]
    return dataset, label_list


class InputFeatures():
    """A single set of features of data."""
    def __init__(self, ntokens, input_ids, mask, segment_ids, label_ids, label_map):
        self.ntokens = ntokens
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_map = label_map


def convert_single_example(example, label_list, max_seq_length, tokenizer):
    """
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [B-PER,I-PER,X,O,O,O,X]
    """
    label_map = {}
    # where start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    textlist = example.text
    labellist = example.label
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length:
        tokens = tokens[0:(max_seq_length)]
        labels = labels[0:(max_seq_length)]
    ntokens = []
    segment_ids = []
    label_ids = []
    # ntokens.append("[CLS]")
    # segment_ids.append(0)
    # label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1] * len(input_ids)
    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(-1)    # !! not zero !!
        ntokens.append("[PAD]")
    feature = InputFeatures(
        ntokens=ntokens,
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        label_map=label_map
    )
    return feature


class Score():
    def __init__(self, tp, fp, tn, fn):
        self.total = tp + fp + tn + fn
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def calc_score(self):
        score = Score(self.tp, self.fp, self.tn, self.fn)
        acc = score.acc()
        rec = score.rec()
        fpr = score.fpr()
        prec = score.prec()
        f1 = score.f1(rec, prec)
        return acc, rec, fpr, prec, f1

    def acc(self):
        try:
            self.acc = (self.tp + self.tn) / self.total
        except ZeroDivisionError:
            self.acc = 0
        return self.acc

    def rec(self):
        try:
            self.rec = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.rec = 0
        return self.rec

    def fpr(self):
        try:
            self.fpr = self.fp / (self.tn + self.fp)
        except ZeroDivisionError:
            self.fpr = 0
        return self.fpr

    def prec(self):
        try:
            self.prec = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.prec = 0
        return self.prec

    def f1(self, rec, prec):
        try:
            self.f1 = (2 * rec * prec) / (rec + prec)
        except ZeroDivisionError:
            self.f1 = 0
        return self.f1
