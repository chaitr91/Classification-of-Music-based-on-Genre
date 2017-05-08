import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import librosa
from helper_util import *

SOUND_SAMPLE_LENGTH = 30000

HAMMING_SIZE = 100
HAMMING_STRIDE = 40

labelsDict = {
    'blues'     :   0,
    'classical' :   1,
    'jazz'      :   2,
    'metal'     :   3,
    'pop'       :   4,
    'rock'      :   5
}


def die_with_usage():
    """ HELP MENU """
    print("USAGE: python predict.py [path to MSD mp3 data]")
    sys.exit(0)

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)



def prepossessingAudio(audioPath, ppFilePath):
    pre, ext = os.path.splitext(audioPath)
    print(pre)
    print(ext)
    data = {}
    if ext == '.mel':
        print("pp input. Skip preprocessing")
        print(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
            pp = pickle.loads(content)
            pp = np.asarray(pp)
            print(pp.shape)


            data[0] = pp
    else:
        print('Preprocessing ' + audioPath)

        melfeaturesArray = []
        mfccfeaturesArray = []
        for i in range(0, SOUND_SAMPLE_LENGTH, HAMMING_STRIDE):
            if i + HAMMING_SIZE <= SOUND_SAMPLE_LENGTH - 1:
                y, sr = librosa.load(audioPath, offset=i / 1000.0, duration=HAMMING_SIZE / 1000.0)

                # Let's make and display a mel-scaled power (energy-squared) spectrogram
                S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

                # Convert to log scale (dB). We'll use the peak power as reference.
                log_S = librosa.logamplitude(S, ref_power=np.max)



                mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
                melfeaturesArray.append(S)

                mfccfeaturesArray.append(mfcc)
                if len(mfccfeaturesArray) == 599:
                    break


        data[0] = melfeaturesArray

    n_input = 599 * 128 * 5
    n_classes = 6
    dropout = 0.75
    learning_rate = 0.01
    train_size = 1250
    
    data = np.asarray([data[i] for i in np.asarray([l for l in data])])
    print("shape")
    print(data.shape[0])
    data = data.reshape((data.shape[0], n_input))

    keep_prob = tf.placeholder(tf.float32)
    # Split Train/Test
    testData = data

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])




    # Construct model
    pred = convolution_network(x, keep_prob)

    classi = tf.argmax(pred,1)
    # Initializing the variables
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Needs model.final in the current directory
    ckpt = tf.train.get_checkpoint_state("./")

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        saver.restore(sess, ckpt.model_checkpoint_path)
        predictions = sess.run(classi, feed_dict={x: testData, keep_prob: 1.})
        print(predictions)
        print(list(labelsDict.keys())[list(labelsDict.values()).index(predictions[0])])

if __name__ == "__main__":

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    i = 0.0
    file_path = sys.argv[1]

    if file_path.endswith('.mp3'):

        filename = os.path.basename(file_path)
        print(filename)
        pre, ext = os.path.splitext(filename)
        ppFileName = pre+".mel"
        print(ppFileName)


    elif file_path.endswith('.mel'):
            print("pickle file")
            ppFileName = file_path

    try:
             prepossessingAudio(file_path, ppFileName)
    except Exception as e:
             print("Error occurred" + str(e))

    if file_path.endswith('au'):
         sys.stdout.write("\r%d%%" % int(i / 7620 * 100))
         sys.stdout.flush()
         i += 1