import numpy as np
from lab2_tools import *
from prondict import prondict


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           - name: phonetic or word symbol corresponding to the model
           - startprob: M+1 array with priori probability of state
           - transmat: (M+1)x(M+1) transition matrix
           - means: MxD array of mean vectors
           - covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """

    concat_hmm = {}
    startprob = np.zeros(hmm1['startprob'].shape[0] + hmm2['startprob'].shape[0] - 1)
    for i in range(startprob.shape[0]):
        if i < hmm1['startprob'].shape[0] - 1:
            startprob[i] = hmm1['startprob'][i]
        else:
            startprob[i] = hmm1['startprob'][-1] * hmm2['startprob'][i - (hmm1['startprob'].shape[0] - 1)]

    transmat = np.zeros((startprob.shape[0], startprob.shape[0]))
    for i in range(transmat.shape[0] - 1):
        for j in range(transmat.shape[1]):
            if i < hmm1['startprob'].shape[0] - 1 and j < hmm1['startprob'].shape[0] - 1:  # Copy of hmm1 values
                transmat[i, j] = hmm1['transmat'][i, j]
            elif i < hmm1['startprob'].shape[0] - 1:  # Product of the last value of hmm1 and values of hmm2
                transmat[i, j] = hmm1['transmat'][i, -1] * hmm2['startprob'][j - (hmm1['startprob'].shape[0] - 1)]
            elif j >= hmm1['startprob'].shape[0] - 1:  # Copy of hmm2 values
                transmat[i, j] = hmm2['transmat'][i - (hmm1['transmat'].shape[0] - 1),
                                                  j - (hmm1['transmat'].shape[0] - 1)]
    transmat[-1, -1] = 1  # Assign value 1 to the last transition

    means = np.vstack((hmm1['means'], hmm2['means']))
    covars = np.vstack((hmm1['covars'], hmm2['covars']))

    concat_hmm['startprob'] = startprob
    concat_hmm['transmat'] = transmat
    concat_hmm['means'] = means
    concat_hmm['covars'] = covars

    return concat_hmm


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           - name: phonetic or word symbol corresponding to the model
           - startprob: M+1 array with priori probability of state
           - transmat: (M+1)x(M+1) transition matrix
           - means: MxD array of mean vectors
           - covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])

    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """


def viterbi(log_emlik, log_startprob, log_transmat, force_final_state=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        force_final_state: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """


def updateMeanAndVar(x, log_gamma, variance_floor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         x: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         variance_floor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """


def main():
    # example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()
    phone_hhms = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
    isolated = {}
    for digit in prondict.keys():
        isolated[digit] = ['sil'] + prondict[digit] + ['sil']
    word_hmms = {'o': concatHMMs(phone_hhms, isolated['o'])}


if __name__ == "__main__":
    main()
