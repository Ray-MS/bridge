cd $(dirname $0)/..
pip install -r requirements.txt

python train.py
sudo sh -c "ps -ef | grep ssh | awk '{print \$2}' | xargs kill -9"
