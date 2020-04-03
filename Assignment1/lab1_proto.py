# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

from Assignment1.lab1_tools import lifter, trfbank
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist as euclidean
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


def mspec(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        samples: list of received samples.
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    enframed_samples = np.zeros((1, winlen))  # Define the resulting matrix of samples
    start = 0  # Define starting pointer of the sample array
    end = min(winlen, len(samples))  # Define ending pointer of the sample array
    enframed_samples[0, start:end] = samples[start:end]  # Enframe the first window of samples
    overlap_samples = enframed_samples[0, winlen - winshift:winlen]  # Define the first overlap of samples
    start = end  # Update starting pointer
    end += min(winlen - winshift, len(samples) - end)  # Update ending pointer
    while end - start + len(overlap_samples) == enframed_samples.shape[1]:  # While the sample fills the window
        enframed_samples = np.vstack((enframed_samples, np.zeros((1, winlen))))  # Generate new window
        enframed_samples[-1, 0:len(overlap_samples)] = overlap_samples
        enframed_samples[-1, len(overlap_samples):end - start + len(overlap_samples)] = samples[start:end]
        overlap_samples = enframed_samples[-1, winlen - winshift:winlen]
        start = end
        end += min(winlen - winshift, len(samples) - end)

    return enframed_samples

    
def preemp(samples, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        samples: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    samples = signal.lfilter([1, -p], [1], samples)  # Apply filter to samples enframed matrix

    return samples


def windowing(samples):
    """
    Applies hamming window to the input frames.

    Args:
        samples: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """

    hamming_window = signal.hamming(samples.shape[1], sym=False)  # Obtain hamming window according to window shape

    return hamming_window * samples  # Return samples once window is applied


def powerSpectrum(sample, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        sample: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """

    f = fft(sample, nfft)
    power_spectrum = np.abs(pow(f, 2))

    return power_spectrum


def logMelSpectrum(sample, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        sample: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """

    trf = trfbank(samplingrate, sample.shape[1])
    _mspec = np.log(sample @ trf.T)

    return _mspec


def cepstrum(sample, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        sample: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """

    cepstral_coeff = dct(sample)[:, :nceps]
    
    return cepstral_coeff


def dtw(x, y, dist=euclidean):
    """Dynamic Time Warping.

    Args:
        x: arrays of size NxD and MxD respectively, where D is the dimensionality
           and N, M are the respective lenghts of the sequences
        y: arrays of size NxD and MxD respectively, where D is the dimensionality
           and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        gl_distance: global distance between the sequences (scalar) normalized to len(x)+len(y)
        lc_dist: local distance between frames from x and y (NxM matrix)
        acc_dist: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough acc_dist

    Note that you only need to define the first output for this exercise.
    """

    lc_dist = dist(x, y)  # Calculation of the local distances
    acc_dist = float('inf') * np.ones(lc_dist.shape)  # Initialization of acc. distances
    acc_dist[0, 0] = lc_dist[0, 0]  # First distance --> nothing accumulated
    for i in range(1, acc_dist.shape[1]):
        acc_dist[0, i] = acc_dist[0, i - 1] + lc_dist[0, i]  # 1st row
    for i in range(1, acc_dist.shape[0]):
        acc_dist[i, 0] = acc_dist[i - 1, 0] + lc_dist[i, 0]  # 1st col

    for i in range(1, acc_dist.shape[0]):
        for j in range(1, acc_dist.shape[1]):
            acc_dist[i, j] = lc_dist[i, j] + min(acc_dist[i - 1, j], acc_dist[i, j - 1], acc_dist[i - 1, j - 1])

    gl_distance = acc_dist[-1, -1] / (x.shape[0] + y.shape[0])

    return gl_distance, lc_dist, acc_dist


def main():
    example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
    samples = example['samples']
    winlen = int(example['samplingrate'] * 0.02)
    winshift = int(example['samplingrate'] * 0.01)

    enframed = enframe(samples, winlen, winshift)
    plt.pcolormesh(enframed)
    plt.show()
    pre_emphasized = preemp(enframed, p=0.97)
    plt.pcolormesh(pre_emphasized)
    plt.show()
    windowed = windowing(pre_emphasized)
    plt.pcolormesh(windowed)
    plt.show()
    _fft = powerSpectrum(windowed, 512)
    plt.pcolormesh(_fft)
    plt.show()
    _mspec = logMelSpectrum(_fft, example['samplingrate'])
    plt.pcolormesh(_mspec)
    plt.show()
    ceps = cepstrum(_mspec, 13)
    plt.pcolormesh(ceps)
    plt.show()


if __name__ == "__main__":
    main()
