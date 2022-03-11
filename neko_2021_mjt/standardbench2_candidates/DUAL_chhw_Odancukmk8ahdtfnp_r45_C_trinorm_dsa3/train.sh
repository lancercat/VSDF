export PYTHONPATH=../../../
export CUDA_VISIBLE_DEVICES=$1
mkdir jtrmodels500 ;python train.py 2>&1  500| tee PLAYDAN500.log
mkdir jtrmodels1000 ;python train.py 2>&1 1000| tee PLAYDAN1000.log
mkdir jtrmodels1500 ;python train.py 2>&1 1500| tee PLAYDAN1500.log
mkdir jtrmodels2000 ;python train.py 2>&1 2000| tee PLAYDAN2000.log