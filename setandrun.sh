virtualenv -p python3 debugenv
mkdir saved
mkdir results
source debugenv/bin/activate 
sudo pip3 install -r requirements.txt
python run_tweet_analyzer.py