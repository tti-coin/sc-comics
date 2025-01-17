local template = import "template.libsonnet";

template.DyGIE {
  // Required "hidden" fields.
  data_paths: {
    train: "data/sccomics/train5.jsonl",
    validation: "data/sccomics/dev5/2.jsonl",
    test: "data/sccomics/dev5/1.jsonl",
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "relation",

  // Optional "hidden" fields
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 0,
  max_span_width: 10,

  // Modify the data loader and trainer.
  // data_loader +: {
  //   batch_size: 5
  // },
   trainer +: {
    cuda_device: 0,
    optimizer +: {
      lr: 5e-4
   }
  },

  // Specify an external vocabulary
  // vocabulary: {
  //   type: "from_files",
  //   directory: "vocab"
  // },
}
