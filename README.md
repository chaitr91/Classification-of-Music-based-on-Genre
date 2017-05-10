# Classification-of-Music-based-on-Genre

Run download.py with Spotify clientId and secretKey to generate music genre's preview songs of 30seconds duration.
 - python3 download.py dowl

Create sub-data folder with only few music genre to train the neural network. We have considered 6.

Run gen_labels.py to genrate 50ms label of 30seconds/50 =600 labels of data fr each song
 - python3 gen_labels.py mel/ labData datalabels

Run pre-process.py to generate the mel
  - python3 preprocess.py finaldata/
  
Run trainmel.py to Train the neural nwk
  - python3 train_mel.py labData datalabels

Run preict.py to predict the genre of music
