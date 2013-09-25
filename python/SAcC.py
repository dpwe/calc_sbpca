#!/usr/bin/env python

"""
Subband Autocorrelation Classification (SAcC) Pitch Tracker feature
Based on Matlab code by Byung Suk Lee and Dan Ellis
Python port based on SRI Feature template.
2013-08-25 Dan Ellis dpwe@ee.columbia.edu
"""

import math
import numpy as np
import scipy.signal
import scipy.io
import scipy.cluster.vq

import mlp
import sbpca

################## from sbpca_viterbi.m
def sbpca_viterbi(posteriors, hmm_vp = 0.9):
    """
    % path = sbpca_viterbi(posteriors, hmm_vp)
    %    Find the best (viterbi) path through a set of pitch class
    %    posteriors, for the SAcC pitch tracker.
    %    <posteriors> is <nbins> x <nframes>
    %    <hmm_vp> is 
    % 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca refactor cleanup
    """

    # Equalizing variance in log-posterior domain per BSL implementation
    sposts = np.exp(standardize(np.log(posteriors)))

    # Set up data for decode
    nbins, nframes = np.shape(sposts)
    npch = nbins - 1  # number of actual pitches (i.e., all except unvoiced)

    # Parameters
    uvtrp = 0.9  # prob of going from unvoiced to voiced (9x larger
                 # than BSL's code, to compensate for normalization of txmat)
    vutrp = 0.01 # prob of going from voiced to unvoiced
    transfloor = np.exp(-10.0)   # smallest transition probability
    wdyn = 3.0   # laplacian half-width for transition probs
    #hmm_vp = 0.9     # scaling of unvoiced state

    # Transition matrix - row = from, column = to
    # A matrix of how far apart two bins are
    ijdiff = np.abs(np.tile(range(npch), (npch,1)).transpose() - range(npch))
    # pitch-to-pitch transitions are laplacian
    # summed in log-domain, per BSL...
    pptxmat = np.log(transfloor + np.exp(np.exp(-np.abs(ijdiff)/wdyn)))
    # normalize rows of pitch-to-pitch transitions to be true probabilities
    pptxmat /= pptxmat.sum(axis=1)[:,np.newaxis]
    # transmat wraps unvoiced state around pitch-to-pitch
    transmat = np.vstack( (np.r_[(1-uvtrp), uvtrp/npch*np.ones(npch)], 
                           np.hstack((vutrp*np.ones( (npch,1) ), (1-vutrp)*pptxmat))))

    # penalize unvoiced posterior & renormalize
    sposts[0,] = hmm_vp * sposts[0,]
    # renormalize columns
    sposts /= sposts.sum(axis=0)
    priors = np.ones(nbins)/nbins

    return viterbi_path(sposts, priors, transmat)

#%%%%%%%%%%%%%%%%%%%%%%%
def standardize(A):
    """
    N = standardize(A)
    % Make each column of an array have a zero mean and unit sd
    % was "normalise" by kslee@ee.columbia.edu (not to confuse with kpm's normalise)
    """
    ndim = np.size(A, axis=0)
    s = A.std(axis=0)
    # normalize each column
    return (A - A.mean(axis=0))/(s+(s==0))
    

################## from viterbi_path.m
def viterbi_path(posteriors, priors, transmat):
    """
    % path = viterbi_path(posteriors, priors, transmat)
    %     Find best path through spectrogram-like posteriors (one
    %     column per time frame).  Transmat is row from, column to.
    %     Linear probabilities (not log). 
    %     Return sequence of state indices.
    % 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca refactor cleanup
    """
    (nbins, nframes) = np.shape(posteriors)

    # Array to hold traceback
    prev = np.zeros( (nbins, nframes) , int);

    # <pstate> holds normalized  probability-to-date of landing in this
    # state along best path 
    pstate = priors*posteriors[:,0]
    # normalize probs of best path to each state, to avoid underflow
    pstate = pstate/np.sum(pstate)

    use_log = True
    #print "use_log=", use_log

    # now calculate forward
    if use_log:
        # log domain
        logtransmat = np.log(transmat.transpose())
        pstate = np.log(pstate)
        for i in range(1, nframes):
            probs = (logtransmat 
                     + np.tile(np.log(posteriors[:,i]),(nbins,1)).transpose() 
                     + np.tile(pstate, (nbins,1)))
            pstate = np.max(probs, axis=1)
            prev[:,i] = np.argmax(probs, axis=1)
            # Renormalize to keep probabilities in a sensible range
            pstate = pstate - np.mean(pstate)

    else:
        # linear likelihood domain
        for i in range(1, nframes):
            # Find most likely combination of previous prob-to-path, and transition
            probs = transmat.transpose() * np.outer(posteriors[:,i], pstate)
            pstate = np.max(probs, axis=1)
            prev[:,i] = np.argmax(probs, axis=1)
            # Renormalize to keep probabilities in a sensible range
            pstate = pstate/sum(pstate)

    # traceback best precedent matrix to get best path
    path = np.zeros(nframes, int)
    # best final state
    path[nframes-1] = np.argmax(pstate)
    # .. and all its predecessors
    for i in range(nframes,1,-1):
        path[i-2] = prev[path[i-1], i-1]
    return path

#####################################

def dithering(x):
    """
    % y = dithering(x)
    %    Add low-level noise to x to avoid digital zeros
    """
    # Ensure consistent random sequence (in dither()
    np.random.seed(0)
    # Generate the dither sequence
    xlen = len(x)
    dither = np.random.rand(xlen) + np.random.rand(xlen) - 1
    # add it on 120 dB below the signal
    spow = np.std(x)
    #print "dithering off"
    #return x
    return x + 1e-6 * spow * dither

# For SRI's wavreading code
import scipy.signal as ss
import scipy.io.wavfile as wav

# Main class

class SAcC(object):
    """ Compute Subband Autocorrelation Classification (SAcC) pitch track
    """
    def __init__(self, config):
        """ Initialize default values """
        self.config = config
        # initialize the sbpca subsystem
        self.sbpca = sbpca.sbpca(config)
        # initialize the mlp subsytem
    	self.net = mlp.mlp(config['wgt_file'], config['norms_file'])
        # parameters specific to SAcC part
    	self.ptchtab = np.r_[0, np.loadtxt(config['pcf_file'])]
    	self.hmm_vp = config['hmm_vp']
        if 'output_pcas' in config:
            self.output_pcas = config['output_pcas']
        else:
            self.output_pcas = False;
        if 'output_autocos' in config:
            self.output_autocos = config['output_autocos']
        else:
            self.output_autocos = False;


    def __call__(self, filename):
        """ This is called for each file """
        sr, w = wav.read(filename)  # remove dependency on libsndfile. Only accept wav, use Sox
        d = np.asfarray(w) / 32768.0  # normalize short ints to floats of -1 / 1
        if sr == 16000 and self.sbpca.sr == 8000:
            # slight trim to ss.decimate to make its phase align to matlab's resample
            d = ss.decimate(np.r_[d[1:],0], 2, ftype='fir')
            delay = 7
            d = np.r_[d[delay:],np.zeros(delay)]
            sr = sr/2
        assert sr == self.sbpca.sr
        # Actually run it
        ftrs =  self.SAcC(d, sr)
        # Return the features
        return ftrs

    def SAcC(self, d, sr):
        """
        Run the SAcC pitch tracker on the specified waveform/sampling rate
        using the configuration specified on construction
        Return two vectors, pitch (in Hz) and P(voicing) (posterior)
        """
        # Pad out d with zeros so get right number of winsamps frames
        # (and add unique dithering noise over whole signal)
        X = dithering(np.r_[d, np.zeros(self.sbpca.maxlags)])
        # Pre-allocate whole activations matrix
        nframes = self.sbpca.nframes(len(d))
        acts = np.zeros( (len(self.net.OB), nframes) )
        (nChs, nDim, nLag) = np.shape(self.sbpca.mapping)
        if self.output_pcas:
            ftrs = np.zeros( (nChs, nDim, nframes) )
        elif self.output_autocos:
            ftrs = np.zeros( (nChs, nLag, nframes) )
        else:
            ftrs = np.zeros( (2, nframes) )

        # How many frames to process each time in loop
        blockframes = 100
        framesamps = self.sbpca.framesamps
        blocksamps = blockframes * framesamps
        nblocks = int(np.ceil(float(nframes) / float(blockframes)))

        # How many frames do we try to prepad?
        prepadframes = 10

        isfirst = 1
        for block in range(nblocks):
            # Figure next block of samples, including pre- and post-padding
            actualprepadframes = min(prepadframes, block*blockframes)
            blockbasesamp = block*blocksamps
            blocklastsamp = min(len(X), blockbasesamp + blocksamps 
                                +self.sbpca.padsamps)
            Xpts = X[(blockbasesamp - actualprepadframes*framesamps):blocklastsamp]
            # Run the sbpca part
            if self.output_autocos:
                ftr = self.sbpca.calc_autocos(Xpts, sr, isfirst)
                (nsb,nlg,nfr) = np.shape(ftr)
                blockix = range(block*blockframes, block*blockframes+(nfr-actualprepadframes))
                ftrs[:,:,blockix] = ftr[:,:,actualprepadframes:]
            else:
                ftr = self.sbpca.calc_sbpcas(Xpts, sr, isfirst)
                (nsb,npc,nfr) = np.shape(ftr)
                blockix = range(block*blockframes, block*blockframes+(nfr-actualprepadframes))
                if self.output_pcas:
                    ftrs[:,:,blockix] = ftr[:,:,actualprepadframes:]
                # Run the MLP classifier
                acts[:,blockix] = self.net.apply(ftr[:,:,actualprepadframes:].transpose().reshape((nfr-actualprepadframes), nsb*npc)).transpose()
            isfirst = 0
        
        if (self.output_autocos == 0 & self.output_pcas == 0):
            # Run viterbi decode on all activations stitched together
            pth      = sbpca_viterbi(acts, self.hmm_vp)
            # first activation is Pr(unvoiced), so Pr(voiced) is its complement
            pvx = 1.0 - acts[0,]
            # Convert pitch bin indices to frequencies in Hz by table lookup
            pitches = self.ptchtab[pth]
            # Return one row of two values per frame
            ftrs = np.c_[pitches, pvx]

    	return ftrs


############## Provide a command-line wrapper

if __name__=="__main__":

    import sys

    if len(sys.argv) != 3:
        raise NameError( ("Usage: ", sys.argv[0], " inputsound.wav outputpitchtrack.txt") )

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
    # mlp params
    #config['wgt_file']    = 'aux/rats_sr8k_bpo6_sb24_k10_aCH_h100.wgt'
    #config['norms_file']  = 'aux/tr_rats_sr8k_bpo6_sb24_k10.norms'
    config['wgt_file']    = 'aux/sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt'
    config['norms_file']  = 'aux/tr_keele_rbf_pinknoise_sr8000_bpo6_nchs24_k10.norms'
    config['pcf_file']    = 'aux/pitch_candidates_freqz.txt'
    # viterbi decode params
    config['hmm_vp']      = 0.9 # interpretation changed c/w Matlab

    # Configure
    sacc_extractor = SAcC(config)

    # Apply
    # d,sr = wavread(inwavfile)
    # pth, pvx = SAcC(d, sr)
    features = sacc_extractor(inwavfile)

    # Write the data out
    nfr = np.size(features, axis=0)
    # Write the data out
#    np.savetxt(outptfile, 
#               np.c_[np.zeros(nfr), range(nfr), features], 
#               fmt='%.6f', delimiter=' ', newline='\n')
    utt = 0
    with open(outptfile, 'w') as opf:
        for i in range(nfr):
            #opf.write('%d %d %f %f\n' % (utt, i, features[i,0], features[i,1]))
            opf.write('%f %f\n' % (features[i,0], features[i,1]))

