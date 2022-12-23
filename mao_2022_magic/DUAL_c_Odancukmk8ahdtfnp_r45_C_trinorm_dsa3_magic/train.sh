export PYTHONPATH=../../../
export CUDA_VISIBLE_DEVICES=$1
cp config.py config.log
mkdir jtrmodels; python3 train.py 2>&1 | tee PLAYDAN.log