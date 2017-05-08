import librosa
import numpy as np
import pickle
import sys
import os

#all clips downloaded from spotify are 30s long
CLIP_SIZE = 30000

# frame size
FRAME_SIZE = 100
# we require 599 features
FRAME_STRIDE = 40

def mel_feature_extraction(input_file_path, output_file_path):

    melfeatureList = []

    for i in range(0, CLIP_SIZE, FRAME_STRIDE):
        if i + FRAME_SIZE < CLIP_SIZE:
            # Since frame size is greater than frame stride each frame will overlap with the previous frame
            x, fs = librosa.load(input_file_path, offset=i / 1000, duration=FRAME_SIZE / 1000)

            # mel-scaled power spectrogram
            ms = librosa.feature.melspectrogram(x, sr=fs, n_mels=128)

            melfeatureList.append(ms)
            if len(melfeatureList) == 599:
                break
                
    f = open(output_file_path, 'wb')
    f.write(pickle.dumps(melfeatureList))
    f.close()



if __name__ == "__main__":

    print("Path :" + sys.argv[1])
    path = sys.argv[1]

    for (dirpath, dirnames, filenames) in os.walk(path):
        count = 0
        for filename in filenames:
            count += 1
            print(filename)
            if filename.endswith('.mp3'):
                file_path = os.path.join(dirpath, filename)
                preprocess_file_name = os.path.join(dirpath, str(count) + ".mel" )
		
                dir = os.path.dirname(os.path.dirname(preprocess_file_name)) 
                dirname = os.path.basename(dir) 
                splitPath = preprocess_file_name.rsplit(dirname, 1)
                preprocess_file_name = "mel".join(splitPath)

                parentDir = os.path.dirname(preprocess_file_name)
    
                if not os.path.exists(parentDir):
                    os.makedirs(parentDir)
                try:
                    mel_feature_extraction(file_path, preprocess_file_name)
                except Exception as e:
                    print("Error:" + str(e))
