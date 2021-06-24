
cp -a  models/sccomics-f*/prediction_test_f*/* results/prediction_test_all/

python src/calc_score.py results/prediction_test_all/  ../data/sccomics/  results/
