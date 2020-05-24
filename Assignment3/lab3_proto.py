import tensorflow as tf
import numpy as np
from lab3_tools import *
import sys
import os
import math
sys.path.append('..')
from Assignment1 import lab1_proto
from Assignment2 import prondict, lab2_proto, lab2_tools
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import np_utils
from keras.models import load_model
import model

PATH = 'Data/'

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phone_symbols = list()
    if addSilence:
        phone_symbols.append('sil')

    for word in wordList:
        pron = pronDict[word]
        for ph in pron:
            phone_symbols.append(ph)
        if addShortPause:
            phone_symbols.append('sp')

    if addSilence:
        phone_symbols.append('sil')

    return phone_symbols


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

    utteranceHMM = lab2_proto.concatHMMs(phoneHMMs, phoneTrans)
    obsloglik = lab2_tools.log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    stateTransviterbiStateTrans = lab2_proto.viterbi(obsloglik, np.log(utteranceHMM['startprob'] + np.finfo('float').eps),
                                           np.log(utteranceHMM['transmat'] + np.finfo('float').eps))[1]
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]
    phoneme_path = [stateTrans[int(idx)] for idx in stateTransviterbiStateTrans]

    return phoneme_path


def dynamic_features(data, feature_name='lmfcc', stack_size=7, dynamic=True):
    final_data = list()
    final_targets = list()
    for item in tqdm(data):
        # Dynamic features for data
        feature_vec = item[feature_name]
        target = item['targets']
        if dynamic:
            dyn_lmfcc = np.zeros((feature_vec.shape[0], stack_size, feature_vec.shape[1]))
            for idx, feature in enumerate(feature_vec):
                dyn_mat_lmfcc = np.zeros((feature.shape[0], stack_size))
                offset = math.ceil(stack_size / 2)
                dyn_mat_lmfcc[0] = np.hstack((feature[np.flip(np.arange(1, offset))], feature[0],
                                              feature[np.arange(1, offset)]))
                offset = int(stack_size / 2)
                for i in range(1, feature.shape[0]):
                    feature = np.roll(feature, -1)
                    dyn_mat_lmfcc[i] = np.hstack((dyn_mat_lmfcc[i - 1][1:], feature[offset]))
                dyn_lmfcc[idx] = np.copy(dyn_mat_lmfcc.T)
            final_data.append(dyn_lmfcc)
        else:
            final_data.append(feature_vec)
        final_targets.append(target)

    return final_data, final_targets


def flatten_data(data):
    return np.vstack([data[i].reshape(data[i].shape[0], np.prod(list(data[i].shape[1:]))) for i in range(len(data))])


def flatten_targets(targets):
    return np.hstack([np.hstack(targets[i]) for i in range(len(targets))])


def target_to_index(target, state_list):
    new_target = np.zeros(target.shape, dtype=int)
    for i in range(target.shape[0]):
        new_target[i] = state_list.index(target[i])

    return new_target


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def main():
    np.random.seed(42)
    phoneHMMs = np.load('../Assignment2/lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    state_list = [ph + '_' + str(idx) for ph in phones for idx in range(nstates[ph])]

    filenames = ['Data/traindata.npz', 'Data/testdata.npz']
    sets = ['tidigits/disc_4.1.1/tidigits/train', 'tidigits/disc_4.2.1/tidigits/test']
    for idx, file_name in enumerate(filenames):
        if not os.path.isfile(file_name):
            data = []
            for root, dirs, files in os.walk(sets[idx]):
                for file in tqdm(files):
                    if file.endswith('.wav'):
                        filename = os.path.join(root, file)
                        samples, samplingrate = loadAudio(filename)
                        lmfcc, mspec = lab1_proto.mfcc(samples)
                        wordTrans = list(path2info(filename)[2])
                        phoneTrans = words2phones(wordTrans, prondict.prondict)
                        targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
                        data.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'targets': targets})
            if file_name == 'traindata.npz':
                np.savez(PATH + file_name, traindata=data)
            elif file_name == 'testdata.npz':
                np.savez(PATH + file_name, testdata=data)

    traindata = np.load('Data/traindata.npz', allow_pickle=True)['traindata']
    women_tracks = list()
    men_tracks = list()
    for tr in traindata:
        if 'woman' in tr['filename']:
            women_tracks.append(tr)
        elif 'man' in tr['filename']:
            men_tracks.append(tr)
    men_tracks = np.array(men_tracks)
    women_tracks = np.array(women_tracks)

    val_size = int(len(traindata) * 0.1)  # Percentage of validation data
    men_pos = np.random.choice(len(men_tracks), int(val_size / 2), replace=False)  # Randomly get men samples
    women_pos = np.random.choice(len(women_tracks), int(val_size / 2), replace=False)  # Randomly get women samples
    men_val = men_tracks[men_pos]  # Get validation men
    women_val = women_tracks[women_pos]  # Get validation women
    men_tracks = np.delete(men_tracks, men_pos)  # Delete validation men from training set
    women_tracks = np.delete(women_tracks, women_pos)  # Delete validation women from training set
    # Get training, validation and testing
    traindata = np.concatenate((men_tracks, women_tracks))
    valdata = np.concatenate((men_val, women_val))
    testdata = np.load('Data/testdata.npz', allow_pickle=True)['testdata']

    dynamic = True
    feature = 'lmfcc'
    if dynamic:
        if not os.path.isfile(PATH + 'dynxtraindata_' + feature + '.npz') or not \
                os.path.isfile(PATH + 'dynytraindata_' + feature + '.npz'):
            x, y = dynamic_features(traindata, feature_name=feature, dynamic=dynamic)
            np.savez(PATH + 'dynxtraindata_' + feature + '.npz', traindata=x)
            np.savez(PATH + 'dynytraindata_' + feature + '.npz', traindata=y)
        x_train = np.load(PATH + 'dynxtraindata_' + feature + '.npz', allow_pickle=True)['traindata']
        y_train = np.load(PATH + 'dynytraindata_' + feature + '.npz', allow_pickle=True)['traindata']
        if not os.path.isfile(PATH + 'dynxvaldata_' + feature + '.npz') or not \
                os.path.isfile(PATH + 'dynyvaldata_' + feature + '.npz'):
            x, y = dynamic_features(valdata, feature_name=feature, dynamic=dynamic)
            np.savez(PATH + 'dynxvaldata_' + feature + '.npz', valdata=x)
            np.savez(PATH + 'dynyvaldata_' + feature + '.npz', valdata=y)
        x_val = np.load(PATH + 'dynxvaldata_' + feature + '.npz', allow_pickle=True)['valdata']
        y_val = np.load(PATH + 'dynyvaldata_' + feature + '.npz', allow_pickle=True)['valdata']
        if not os.path.isfile(PATH + 'dynxtestdata_' + feature + '.npz') or not \
                os.path.isfile(PATH + 'dynytestdata_' + feature + '.npz'):
            x, y = dynamic_features(testdata, feature_name=feature, dynamic=dynamic)
            np.savez(PATH + 'dynxtestdata_' + feature + '.npz', testdata=x)
            np.savez(PATH + 'dynytestdata_' + feature + '.npz', testdata=y)
        x_test = np.load(PATH + 'dynxtestdata_' + feature + '.npz', allow_pickle=True)['testdata']
        y_test = np.load(PATH + 'dynytestdata_' + feature + '.npz', allow_pickle=True)['testdata']
    else:
        if not os.path.isfile(PATH + 'xtraindata_' + feature + '.npz') or not \
                os.path.isfile(PATH + 'ytraindata_' + feature + '.npz'):
            x, y = dynamic_features(traindata, feature_name=feature, dynamic=dynamic)
            np.savez(PATH + 'xtraindata_' + feature + '.npz', traindata=x)
            np.savez(PATH + 'ytraindata_' + feature + '.npz', traindata=y)
        x_train = np.load(PATH + 'xtraindata_' + feature + '.npz', allow_pickle=True)['traindata']
        y_train = np.load(PATH + 'ytraindata_' + feature + '.npz', allow_pickle=True)['traindata']
        if not os.path.isfile(PATH + 'xvaldata_' + feature + '.npz') or not \
                os.path.isfile(PATH + 'yvaldata_' + feature + '.npz'):
            x, y = dynamic_features(valdata, feature_name=feature, dynamic=dynamic)
            np.savez(PATH + 'xvaldata_' + feature + '.npz', valdata=x)
            np.savez(PATH + 'yvaldata_' + feature + '.npz', valdata=y)
        x_val = np.load(PATH + 'xvaldata_' + feature + '.npz', allow_pickle=True)['valdata']
        y_val = np.load(PATH + 'yvaldata_' + feature + '.npz', allow_pickle=True)['valdata']
        if not os.path.isfile(PATH + 'xtestdata_' + feature + '.npz') or not \
                os.path.isfile(PATH + 'ytestdata_' + feature + '.npz'):
            x, y = dynamic_features(testdata, feature_name=feature, dynamic=dynamic)
            np.savez(PATH + 'xtestdata_' + feature + '.npz', testdata=x)
            np.savez(PATH + 'ytestdata_' + feature + '.npz', testdata=y)
        x_test = np.load(PATH + 'xtestdata_' + feature + '.npz', allow_pickle=True)['testdata']
        y_test = np.load(PATH + 'ytestdata_' + feature + '.npz', allow_pickle=True)['testdata']

    # Flatten into matrix
    x_train = flatten_data(x_train)
    x_val = flatten_data(x_val)
    x_test = flatten_data(x_test)
    y_train = flatten_targets(y_train)
    y_val = flatten_targets(y_val)
    y_test = flatten_targets(y_test)
    # Normalize data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype('float32')
    x_val = scaler.transform(x_val).astype('float32')
    x_test = scaler.transform(x_test).astype('float32')
    output_dim = len(state_list)
    y_train = target_to_index(y_train, state_list)
    y_val = target_to_index(y_val, state_list)
    y_test = target_to_index(y_test, state_list)
    y_train = np_utils.to_categorical(y_train, output_dim)
    y_val = np_utils.to_categorical(y_val, output_dim)
    y_test = np_utils.to_categorical(y_test, output_dim)

    # Model train
    if dynamic:
        if not os.path.isfile('model_' + feature + 'dynamic' + '.h5'):
            classifier = model.classifier(x_train[0].shape, output_dim)
            classifier.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            classifier.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=256, epochs=10)
            classifier.save('model_' + feature + 'dynamic' + '.h5')
        else:
            classifier = tf.keras.models.load_model('model_' + feature + 'dynamic' + '.h5')
    else:
        if not os.path.isfile('model_' + feature + '.h5'):
            classifier = model.classifier(x_train[0].shape, output_dim)
            classifier.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
            classifier.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=256, epochs=1)
            classifier.save('model_' + feature + '.h5')
        else:
            classifier = tf.keras.models.load_model('model_' + feature + '.h5')

    # Model predict
    prediction = classifier.evaluate(x_test, y=y_test, batch_size=256)
    print("Loss: " + str(prediction[0]) + "\tAccuracy: " + str(prediction[1]))
    
    y_pred = model.predict(xtest)

    Y_pred = np.argmax(y_pred, axis = 1) 
    Y_true = np.argmax(y_test, axis = 1) 

    confusion_mtx  = confusion_matrix(Y_true, Y_pred) 

    plot_confusion_matrix(confusion_mtx, classes = list(len(state_list)))


if __name__ == "__main__":
    main()
