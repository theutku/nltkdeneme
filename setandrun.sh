virtualenv -p python3 debugenv
source debugenv/bin/activate 
sudo pip3 install -r requirements.txt
python test.py