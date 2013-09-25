#!/usr/bin/env python

"""
Auditory-model features based on subband PCA (sbpca) faetures
Based on calcAudSemaFtrs matlab code.
2013-09-19 Dan Ellis dpwe@ee.columbia.edu
"""

import math
import numpy as np
import scipy.signal
import scipy.io

import mlp
import sbpca

################### from spca_vqs.m

def sbpca_vqs(pcas, vqcodebook, vqmeans, vqstds):
    """
    % vqs = sbpca_vqs(pcas, params)
    %    Convert subband principal components to vector quantized values 
    %    for SBPCA.
    %    pcas are (pcadim == 10, chans == 24, timefrm)
    %    vqs are (blocks == 4, timefrm)
    %    each value is one VQ index (0..1023).
    % 2013-08-03 Dan Ellis dpwe@ee.columbia.edu
    """

    (nRec, nFtr, K) = np.shape(vqcodebook) # 4 x 1000 x 60 
    (nChs, nDim, nTim) = np.shape(pcas)    # 24 x 10 x T

    # Reshape pcas into vectors to be quantized
    grouping = int(nChs / nRec) # better hope it's 24 / 4 = 6

    # allocate
    vqs = np.zeros( (nRec, nTim), int)

    for i in range(nRec):
        chs = range(i*grouping, (i+1)*grouping)
        # 60 dim codewords in Matlab file are ordered with the 6 chans as 
        # the fastest rotating dimension, so make sure python array does this
        # too before collapsing 10x6 into rows of 60
        vqs[i,] = acVQQuantize(pcas[chs,:,:].transpose(1,0,2).reshape(grouping*nDim, nTim).transpose(), 
                                vqcodebook[i,], vqmeans[i,], vqstds[i,])

    return vqs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def acVQQuantize(D, vqcodes, vqmean, vqstd):
    """
    % I = acVQQuantize(D,CB)
    %    Return I as the quantization of each data row D according to
    %    the codebook CB.mean, CB.std, CB.codewords.
    % 2012-07-31 Dan Ellis dpwe@ee.columbia.edu
    """
    (nTim, nD) = np.shape(D)
    maxdrows = min(10000, nTim)
    nblocks = int(math.ceil(nTim/maxdrows))

    I = np.zeros( nTim, int)

    if nD > 0:
        for b in range(nblocks):
            bkix = range(b*maxdrows, min(nTim, (b+1)*maxdrows))
            # Extract rows and apply mean/var normalization
            Dblock = (D[bkix,] - vqmean) / vqstd

            # Quantize
            I[bkix] = scipy.cluster.vq.vq(Dblock, vqcodes)

    return I


#####################################

# Main class

class AudFtr(object):
    """ 
    Compute sbpca-based auditory features
    meaning 4000-dim histograms over 2 sec windows.
    """
    def __init__(self, config):
        """ Initialize default values """
        self.config = config
        # initialize sbpca subsystem
        self.sbpca = sbpca.sbpca(config)
        # set up AudFtr-specific params
        # Read in the VQ codebooks
    	C = scipy.io.loadmat(config['vq_file'])
        # Read the codebook as rects x codewords x dims (4 x 1000 x 60)
    	self.vqcodebook = C["codebook"].transpose( (0,2,1) )
        self.vqmeans = C["recMean"]
        self.vqstds  = C["recStd"]

    def __call__(self, d, sr):
        """
        Run the sbpcahist ftr extractor on the specified waveform/sampling rate
        using the configuration specified on construction
        Return a matrix of <ftrs> x <blocks>
        """
        # Calculate the subband PCA features
        pcas     = self.sbpca.calc_sbpcas(d, sr)
        # Vector quantize in individual frames
        vqs      = sbpca_vqs(pcas, self.vqcodebook, self.vqmeans, self.vqstds)
        # Collapse blocks into histograms
        #hists    = sbpca_hist(vqs)

        ftrs = vqs

    	return ftrs


############## Provide a command-line wrapper

if __name__=="__main__":

    import sys
    # For SRI's wavreading code
    import scipy.signal as ss
    import scipy.io.wavfile as wav


    if len(sys.argv) != 3:
        raise NameError( ("Usage: ", sys.argv[0], " inputsound.wav outputaudftr.txt") )

    inwavfile = sys.argv[1]
    outptfile = sys.argv[2]

    # Setup config
    config = {}
    # sbpca params
    config['pca_file']    = 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat' # diff file for py
    config['nchs']        = 24
    config['SBF_sr']      = 8000
    config['SBF_fmin']    = 100
    config['SBF_bpo']     = 6
    config['SBF_q']       = 8  # not actually used for SlanPat ERB filters
    config['SBF_order']   = 2  # not actually used for SlanPat ERB filters
    config['ac_win']      = 0.025  # New option in Python - autoco window len
    config['ac_hop']      = 0.010  # New option in Python - autoco hop
    # audftr params
    config['vq_file']     = '../CB-sbpca-4x60x1000.mat'

    # Configure
    ftr_extractor = AudFtr(config)

    # Read in wav file
    sr, w = wav.read(inwavfile)
    d = np.asfarray(w) / 32768.0  # normalize short ints to floats of -1 / 1

    # Apply
    features = ftr_extractor(d, sr)

    # Write the data out
    nfr = np.size(features, axis=0)
    # Write the data out
    np.savetxt(outptfile, 
               np.c_[np.zeros(nfr), range(nfr), features], 
               fmt='%.6f', delimiter=' ', newline='\n')

