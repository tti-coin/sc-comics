echo "Fold 1"
mkdir -p models/sccomics-f1/prediction_test_f1/
cp -a ../data/sccomics/5-fold_CV/dev1/1/*.txt models/sccomics-f1/prediction_test_f1/
python src/dygiepp2ann.py models/sccomics-f1/prediction_test_f1.jsonl

echo "Fold 2"
mkdir -p models/sccomics-f2/prediction_test_f2/
cp -a ../data/sccomics/5-fold_CV/dev1/2/*.txt models/sccomics-f2/prediction_test_f2/
python src/dygiepp2ann.py models/sccomics-f2/prediction_test_f2.jsonl

echo "Fold 3"
mkdir -p models/sccomics-f3/prediction_test_f3/
cp -a ../data/sccomics/5-fold_CV/dev2/1/*.txt models/sccomics-f3/prediction_test_f3/
python src/dygiepp2ann.py models/sccomics-f3/prediction_test_f3.jsonl

echo "Fold 4"
mkdir -p models/sccomics-f4/prediction_test_f4/
cp -a ../data/sccomics/5-fold_CV/dev2/2/*.txt models/sccomics-f4/prediction_test_f4/
python src/dygiepp2ann.py models/sccomics-f4/prediction_test_f4.jsonl

echo "Fold 5"
mkdir -p models/sccomics-f5/prediction_test_f5/
cp -a ../data/sccomics/5-fold_CV/dev3/1/*.txt models/sccomics-f5/prediction_test_f5/
python src/dygiepp2ann.py models/sccomics-f5/prediction_test_f5.jsonl

echo "Fold 6"
mkdir -p models/sccomics-f6/prediction_test_f6/
cp -a ../data/sccomics/5-fold_CV/dev3/2/*.txt models/sccomics-f6/prediction_test_f6/
python src/dygiepp2ann.py models/sccomics-f6/prediction_test_f6.jsonl

echo "Fold 7"
mkdir -p models/sccomics-f7/prediction_test_f7/
cp -a ../data/sccomics/5-fold_CV/dev4/1/*.txt models/sccomics-f7/prediction_test_f7/
python src/dygiepp2ann.py models/sccomics-f7/prediction_test_f7.jsonl

echo "Fold 8"
mkdir -p models/sccomics-f8/prediction_test_f8/
cp -a ../data/sccomics/5-fold_CV/dev4/2/*.txt models/sccomics-f8/prediction_test_f8/
python src/dygiepp2ann.py models/sccomics-f8/prediction_test_f8.jsonl

echo "Fold 9"
mkdir -p models/sccomics-f9/prediction_test_f9/
cp -a ../data/sccomics/5-fold_CV/dev5/1/*.txt models/sccomics-f9/prediction_test_f9/
python src/dygiepp2ann.py models/sccomics-f9/prediction_test_f9.jsonl

echo "Fold 10"
mkdir -p models/sccomics-f10/prediction_test_f10/
cp -a ../data/sccomics/5-fold_CV/dev5/2/*.txt models/sccomics-f10/prediction_test_f10/
python src/dygiepp2ann.py models/sccomics-f10/prediction_test_f10.jsonl
