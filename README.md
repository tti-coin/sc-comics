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

## Named entity/Relation/Event Extraction by DyGIE++
Official GitHub repository of DyGIE++: https://github.com/dwadden/dygiepp

Our experiments are based on author's official implementation. If you want to reproduce our results, please refer to the original repository above.

The source code for data format conversion and the evaluation program are provided by us. The training configuration file used in our experiments are also available.
### Data format conversion from ann to jsonl
```fish
cd ./dygiepp/src
python3 ann2dygiepp.py source_dir target_dir dataset --main
```
- source_dir: Directory path where the ann files are placed
- target_dir: Directory path where the jsonl files will be saved
- dataset: dataset name
- --main: Specify to distinguish between the named entity classes "Main" and "Element".
### Data format conversion from jsonl to ann
```fish
cd ./dygiepp/src
python3 dygiepp2ann.py source_path
```
- source_path: Path to the jsonl file to be converted

\* Before executing the above command, the save directory must be created and the text files corresponding to the jsonl files must be placed in that directory.
## Main Material Identification (MMI)
```fish
cd ./main_clf
```
### Training
1. Data format conversion from ann to conll.
```fish
cd ./src/data_process
python3 ann2conll.py source_dir target_dir
```
- source_dir: Directory path where the ann files are placed
- target_dir: Directory path where the conll files will be saved.
2. Traning configurelation
- Edit the configuration file below.
```fish
cd ../
vim ./config/train.conf
```
3. Training
```fish
env CUDA_VISIBLE_DEVICES=0 python3 train.py
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
env CUDA_VISIBLE_DEVICES=0 python3 test.py
>>
acc: 0.9427184466019417
rec: 0.7014925373134329
prec: 0.831858407079646
f1: 0.7611336032388664
```
3. Data format conversion from conll to ann
```fish
cd ./data_process
python3 conll2ann.py conll_dir text_dir
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
python3 merge_ann.py dygiepp_dir mmi_dir save_dir
```
- dygiepp_dir: Directory path where the ann files of the predictions made by DyGIE++ are placed
- mmi_dir: Directory path where the ann files of the predictions made by MMI model are placed
- save_dir: Directory path to save the integrated ann files
### Rule based slot extraction
```fish
cd ./src
python3 ann2json.py data_name data_dir save_dir
```
- data_name: dataset name
- data_dir: Directory path where the integrated ann/text files are placed
- save_dir: Directory path to save the json files
