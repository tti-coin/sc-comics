echo "Fold 1"
python src/calc_score.py models/sccomics-f1/prediction_test_f1/ ../data/sccomics/5-fold_CV/dev1/1/ results/

echo "Fold 2"
python src/calc_score.py models/sccomics-f2/prediction_test_f2/ ../data/sccomics/5-fold_CV/dev1/2/ results/

echo "Fold 3"
python src/calc_score.py models/sccomics-f3/prediction_test_f3/ ../data/sccomics/5-fold_CV/dev2/1/ results/

echo "Fold 4"
python src/calc_score.py models/sccomics-f4/prediction_test_f4/ ../data/sccomics/5-fold_CV/dev2/2/ results/

echo "Fold 5"
python src/calc_score.py models/sccomics-f5/prediction_test_f5/ ../data/sccomics/5-fold_CV/dev3/1/ results/

echo "Fold 6"
python src/calc_score.py models/sccomics-f6/prediction_test_f6/ ../data/sccomics/5-fold_CV/dev3/2/ results/

echo "Fold 7"
python src/calc_score.py models/sccomics-f7/prediction_test_f7/ ../data/sccomics/5-fold_CV/dev4/1/ results/

echo "Fold 8"
python src/calc_score.py models/sccomics-f8/prediction_test_f8/ ../data/sccomics/5-fold_CV/dev4/2/ results/

echo "Fold 9"
python src/calc_score.py models/sccomics-f9/prediction_test_f9/ ../data/sccomics/5-fold_CV/dev5/1/ results/

echo "Fold 10"
python src/calc_score.py models/sccomics-f10/prediction_test_f10/ ../data/sccomics/5-fold_CV/dev5/2/ results/
