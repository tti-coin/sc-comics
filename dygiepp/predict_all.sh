echo "Fold 1"
allennlp predict models/sccomics-f1/model.tar.gz data/sccomics/dev1/1.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f1/prediction_test_f1.jsonl --silent

echo "Fold 2"
allennlp predict models/sccomics-f2/model.tar.gz data/sccomics/dev1/2.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f2/prediction_test_f2.jsonl --silent

echo "Fold 3"
allennlp predict models/sccomics-f3/model.tar.gz data/sccomics/dev2/1.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f3/prediction_test_f3.jsonl --silent

echo "Fold 4"
allennlp predict models/sccomics-f4/model.tar.gz data/sccomics/dev2/2.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f4/prediction_test_f4.jsonl --silent

echo "Fold 5"
allennlp predict models/sccomics-f5/model.tar.gz data/sccomics/dev3/1.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f5/prediction_test_f5.jsonl --silent

echo "Fold 6"
allennlp predict models/sccomics-f6/model.tar.gz data/sccomics/dev3/2.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f6/prediction_test_f6.jsonl --silent

echo "Fold 7"
allennlp predict models/sccomics-f7/model.tar.gz data/sccomics/dev4/1.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f7/prediction_test_f7.jsonl --silent

echo "Fold 8"
allennlp predict models/sccomics-f8/model.tar.gz data/sccomics/dev4/2.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f8/prediction_test_f8.jsonl --silent

echo "Fold 9"
allennlp predict models/sccomics-f9/model.tar.gz data/sccomics/dev5/1.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f9/prediction_test_f9.jsonl --silent

echo "Fold 10"
allennlp predict models/sccomics-f10/model.tar.gz data/sccomics/dev5/2.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f10/prediction_test_f10.jsonl --silent
