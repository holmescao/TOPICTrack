exp=bee24_test
dataset=BEE24 # optional: MOT17, MOT20, DANCE, GMOT, BEE24

python3 external/TrackEval/scripts/run_mot_challenge.py \
        --SPLIT_TO_EVAL val \
        --METRICS HOTA Identity CLEAR \
        --TRACKERS_TO_EVAL ${exp} \
        --GT_FOLDER results/gt/ \
        --TRACKERS_FOLDER results/trackers/ \
        --BENCHMARK $dataset
      
python3 external/TrackEval/scripts/run_mot_challenge.py \
        --SPLIT_TO_EVAL val \
        --METRICS HOTA Identity CLEAR \
        --TRACKERS_TO_EVAL ${exp}_post \
        --GT_FOLDER results/gt/ \
        --TRACKERS_FOLDER results/trackers/ \
        --BENCHMARK $dataset