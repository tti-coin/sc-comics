# SC-CoMIcs: A Superconductivity Corpus for Materials Informatics
<a href="https://zenodo.org/badge/latestdoi/343638726"><img src="https://zenodo.org/badge/343638726.svg" alt="DOI"></a>
## Environment
- python 3.7.3
- CUDA 10.1
```fish
pip install -r requirements.txt
# If installation of PyTorch fails, please retry the command below.
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# Install "en_core_sci_sm" of ScispaCy.
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz
```
(\* Virtual environment such as pyenv is recommended.)

### Data download
Download text files (1000 abstracts) from [https://data.mendeley.com/datasets/xc9fjz2p3h/2].
Abstracts can be download for the research purpose under CC BY-NC 3.0.
Then, copy the 1000 text files into sc-comics/data/sccomics where corresponding .ann files are already there.

## Named entity/Relation/Event Extraction by DyGIE++
Official GitHub repository of DyGIE++: https://github.com/dwadden/dygiepp

Our experiments are based on author's official implementation. If you want to reproduce our results, please refer to the original repository above.

The source code for data format conversion and the evaluation program are provided by us. The training configuration file used in our experiments are also available.
### Data format conversion from ann to jsonl
```fish
cd ./dygiepp/src
python ann2dygiepp.py source_dir target_dir dataset --main
```
- source_dir: Directory path where the ann files are placed
- target_dir: Directory path where the jsonl files will be saved
- dataset: dataset name
- --main: Specify to distinguish between the named entity classes "Main" and "Element".

Example:
```fish
python ann2dygiepp.py ../../data/sccomics/5-fold_CV/ ../data/sccomics/ sc-wo-main
```

Here, for some reason, the directory name is "5-fold_CV" but actually, we are doing 10-fold cross validation. This is a bit confusing but there is a good reason behind this. You see the reason in the test section.


For the training with DyGIE++, we do not distinguish "Main" from "Element". "--main" option should be specified when you generate data for the Main Material Identification.

Example:
```fish
python ann2dygiepp.py ../../data/sccomics/5-fold_CV/ ../../data/main_clf/5-fold_CV/ sc-w-main --main
```

### Training with DyGIE++

For the 10-fold cross-validation, we defined 10 configuration files in the training_config directory.
For exmaple, sccomics-f1.jsonnet specifies dev1/1.jsonl (#001-#100) for test, dev1/2.jsonl (#101-#200) for development, and
train1.jsonl (remaining 800 data) for training.

To train for testing on Fold 1,

```fish
bash scripts/train.sh sccomics-f1
```
Note that, DiGIE++ returns an error if the model directory named "sccomics-f1" already exists in the models directory.
When you encounter this error, change the existing model directory or remode if not neccesarry.

### See test data scores with allennlp
Here, the test dataset for 10-fold CV are named as Fold 1=dev1/1.jsonl, Fold 2=dev1/2.jsonl, Fold 3=dev2/1.jsonl, Fold 4=dev2/2.jsonl, Fold 5=dev3/1.jsonl, Fold 6=dev2/2.jsonl, Fold 7=dev4/1.jsonl, Fold 8=dev4/2.jsonl, Fold 9=dev5/1.jsonl, Fold 10=dev5/2.jsonl.


To evaluate on a fold (here, Fold 1) test data, we use an allennlp evaluate command.
```fish
allennlp evaluate models/sccomics-f1/model.tar.gz data/sccomics/dev1/1.jsonl --cuda-device 0 --include-package dygie --output-file models/sccomics-f1/metrics_test_f1.json
```
### Prediction with allennlp

To generate prediction output in the DyGIE format, we use an allennlp predict command.
```fish
allennlp predict models/sccomics-f1/model.tar.gz data/sccomics/dev1/1.jsonl --predictor dygie --cuda-device 0 --include-package dygie --use-dataset-reader --output-file models/sccomics-f1/prediction_test_f1.jsonl --silent
```

### Data format conversion from jsonl to ann
```fish
cd src/
python dygiepp2ann.py source_path
```
- source_path: Path to the jsonl file to be converted

\* Before executing the above command, the save directory must be created and the text files corresponding to the jsonl files must be placed in that directory.

Examples:

Create a save directry with the stem of the prediction file.
```fish
mkdir -p ../models/sccomics-f1/prediction_test_f1
```

Copy text files (i.e. abstracts) to the save directory.
```fish
cp -a ../data/sccomics/5-fold_CV/dev1/1/*.txt models/sccomics-f1/prediction_test_f1/
```

```fish
cd src/
python dygiepp2ann.py ../models/sccomics-f1/prediction_test_f1.jsonl
```

## Calculate detailed scores

Command:
```fish
python calc_score.py prediction_ann_dir gold_ann_dir result_dir
```

Each fold:
```fish
python calc_score.py ../models/sccomics-f1/prediction_test_f1/ ../../data/sccomics/5-fold_CV/dev1/1 ../models/sccomics-f1/
```

Total score:
After conducting the training-prediction steps for all the folds.
```fish
mkdir ../results
mkdir ../results/prediction_test_all
cp -a  ../models/sccomics-f*/prediction_test_f*/* ../results/prediction_test_all/
python calc_score.py ../results/prediction_test_all/  ../../data/sccomics/  ../results/
```

## Main Material Identification (MMI)
```fish
cd ./main_clf
```
### Training
1. Data format conversion from ann to conll.
```fish
cd ./src/data_process
python ann2conll.py source_dir target_dir
```
- source_dir: Directory path where the ann files are placed
- target_dir: Directory path where the conll files will be saved.

Example:

```fish
python ann2conll.py ../../../data/sccomics/ ../../../data/main_clf/
```

2. Traning configurelation
- Edit the configuration file below.
```fish
cd ../
vim ./config/train.conf
```
3. Training
```fish
env CUDA_VISIBLE_DEVICES=0 python train.py train.conf
```
### Test
1. Test configuration
- Edit the configuration file below.
```fish
cd ./src
vim ./config/test.conf
```
2. Prediction and Evaluation
- Accuracy, Recall, Precision and F1 score are calculated.
- The result will be output as "correct" directory if all classification in the abstract are successful, or as "incorrect" if they are not.
```fish
env CUDA_VISIBLE_DEVICES=0 python test.py test.conf
>>
acc: 0.9427184466019417
rec: 0.7014925373134329
prec: 0.831858407079646
f1: 0.7611336032388664
```
3. Data format conversion from conll to ann
```fish
cd ./data_process
python conll2ann.py conll_dir text_dir
```

- conll_dir: Directory path where the conll files are placed
- text_dir: Directory path where the text files corresponding to the conll files are placed
## Slot Extraction
```fish
cd ./slot_ext
```
### Integration of prediction results from DyGIE++ and MMI model
```fish
cd ./src
python merge_ann.py dygiepp_dir mmi_dir save_dir
```
- dygiepp_dir: Directory path where the ann files of the predictions made by DyGIE++ are placed
- mmi_dir: Directory path where the ann files of the predictions made by MMI model are placed
- save_dir: Directory path to save the integrated ann files
### Rule based slot extraction
```fish
cd ./src
python ann2json.py data_name data_dir save_dir
```
- data_name: dataset name
- data_dir: Directory path where the integrated ann/text files are placed
- save_dir: Directory path to save the json files
