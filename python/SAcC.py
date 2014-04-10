#!/usr/bin/env python

"""
Subband Autocorrelation Classification (SAcC) Pitch Tracker feature
Based on Matlab code by Byung Suk Lee and Dan Ellis
Python port based on SRI Feature template.
2013-08-25 Dan Ellis dpwe@ee.columbia.edu
"""

import numpy as np
import scipy.signal
import scipy.io
import scipy.cluster.vq
# For SRI's wavreading code
import scipy.io.wavfile as wav

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
    ijdiff = np.abs(np.tile(range(npch), (npch, 1)).transpose() - range(npch))
    # pitch-to-pitch transitions are laplacian
    # summed in log-domain, per BSL...
    pptxmat = np.log(transfloor + np.exp(np.exp(-np.abs(ijdiff)/wdyn)))
    # normalize rows of pitch-to-pitch transitions to be true probabilities
    pptxmat /= pptxmat.sum(axis=1)[:, np.newaxis]
    # transmat wraps unvoiced state around pitch-to-pitch
    transmat = np.vstack( (np.r_[(1-uvtrp), uvtrp/npch*np.ones(npch)], 
                           np.hstack((vutrp*np.ones( (npch, 1) ), 
                                      (1-vutrp)*pptxmat))))

    # penalize unvoiced posterior & renormalize
    sposts[0,] = hmm_vp * sposts[0,]
    # renormalize columns
    sposts /= sposts.sum(axis=0)
    priors = np.ones(nbins)/nbins

    return viterbi_path(sposts, priors, transmat)

#%%%%%%%%%%%%%%%%%%%%%%%
def standardize(array):
    """
    N = standardize(array)
    % Make each column of an array have a zero mean and unit sd
    % was "normalise" by kslee@ee.columbia.edu (not to confuse with kpm's normalise)
    """
    stddev = array.std(axis=0)
    # normalize each column
    return (array - array.mean(axis=0))/(stddev+(stddev==0))
    

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
    prev = np.zeros( (nbins, nframes) , int)

    # <pstate> holds normalized  probability-to-date of landing in this
    # state along best path 
    pstate = priors*posteriors[:, 0]
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
                     + np.tile(np.log(posteriors[:, i]),(nbins, 1)).transpose() 
                     + np.tile(pstate, (nbins, 1)))
            pstate = np.max(probs, axis=1)
            prev[:, i] = np.argmax(probs, axis=1)
            # Renormalize to keep probabilities in a sensible range
            pstate = pstate - np.mean(pstate)

    else:
        # linear likelihood domain
        for i in range(1, nframes):
            # Find most likely combination of previous prob-to-path, 
            # and transition
            probs = transmat.transpose() * np.outer(posteriors[:, i], pstate)
            pstate = np.max(probs, axis=1)
            prev[:, i] = np.argmax(probs, axis=1)
            # Renormalize to keep probabilities in a sensible range
            pstate = pstate/sum(pstate)

    # traceback best precedent matrix to get best path
    path = np.zeros(nframes, int)
    # best final state
    path[nframes-1] = np.argmax(pstate)
    # .. and all its predecessors
    for pth in range(nframes, 1, -1):
        path[pth-2] = prev[path[pth-1], pth-1]
    return path

#####################################

def dithering(data, noiselevel=1e-3):
    """
    % y = dithering(x, noiselevel)
    %    Add low-level noise to x to avoid digital zeros
    %    noiselevel is scaling factor below SD of signal at which 
    %    noise is added (default 1e-3).
    """
    # Ensure consistent random sequence (in dither()
    np.random.seed(0)
    # Generate the dither sequence
    xlen = len(data)
    dither = np.random.rand(xlen) + np.random.rand(xlen) - 1
    # add it on 120 dB below the signal
    spow = np.std(data)
    #print "dithering off"
    #return x
    #print "dithering at 1e-3"
    #return data + 1e-6 * spow * dither
    return data + noiselevel * spow * dither

# For SRI's wavreading code
import scipy.io.wavfile as wav
from scikits.audiolab import Sndfile
# For command line
import os
import sys

def readsph(filename):
    """ read in audio data from a sphere file.  Return d, sr """
    f = Sndfile(filename, 'r')
    data = f.read_frames(f.nframes, dtype=np.float32)
    sr = f.samplerate
    return data, sr

def readwav(filename):
    """ read in audio data from a wav file.  Return d, sr """
    # Read in wav file
    sr, wavd = wav.read(filename)
    # normalize short ints to floats of -1 / 1
    data = np.asfarray(wavd) / 32768.0  
    return data, sr

def audioread(filename, targetsr=None):
    """
    Read a soundfile of either WAV or SPH, based on filename
    returns d, sr
    """
    fileName, fileExtension = os.path.splitext(filename)
    if fileExtension == ".wav":
        data, sr = readwav(filename)
    elif fileExtension == ".sph":
        data, sr = readsph(filename)
    else:
        raise NameError( ("Cannot determine type of infile " +
                          filename) )
    # Maybe fix sample rate
    #if srate == 16000 and self.sbpca.srate == 8000:
    if targetsr != None and sr != targetsr:
        # Right now, only downsample by integer numbers
        decimfact = int(np.round(sr/targetsr))
        data = scipy.signal.decimate(np.r_[data[1:], 0], 
                                     decimfact, ftype='fir')
        # slight trim to ss.decimate to make its phase align 
        # to matlab's resample 
        # for case of resampling 16 kHz down to 8 kHz
        delay = 7
        data = np.r_[data[delay:], np.zeros(delay)]
        sr = sr/decimfact

    return data, sr


# Main class

class SAcC(object):
    """ Compute Subband Autocorrelation Classification (SAcC) pitch track
    """
    def __init__(self, config):
        """ Initialize default values """
        #self.config = config
        # initialize the sbpca subsystem
        self.sbpca = sbpca.SbPca(config)
        # initialize the mlp subsytem
        self.net = mlp.MLP(config['wgt_file'], config['norms_file'])
        # parameters specific to SAcC part
        self.ptchtab = np.r_[0, np.loadtxt(config['pcf_file'])]
        self.hmm_vp = config['hmm_vp']
        self.n_s = 10.0
        self.start_utt = 0
        self.write_rownum = False
        self.write_time = False
        self.write_sbac = False
        self.write_sbpca = False
        self.write_posteriors = False
        self.write_pitch = True
        self.write_pvx = True
        self.dither_level = 1e-3
        if 'n_s' in config:
            self.n_s = config['n_s']
        if 'start_utt' in config:
            self.start_utt = config['start_utt']
        if 'write_rownum' in config:
            self.write_rownum = config['write_rownum']
        if 'write_time' in config:
            self.write_time = config['write_time']
        if 'write_sbac' in config:
            self.write_sbac = config['write_sbac']
        if 'write_sbpca' in config:
            self.write_sbpca = config['write_sbpca']
        if 'write_posteriors' in config:
            self.write_posteriors = config['write_posteriors']
        if 'write_pitch' in config:
            self.write_pitch = config['write_pitch']
        if 'write_pvx' in config:
            self.write_pvx = config['write_pvx']
        # added 2014-04-10
        if 'dither_level' in config:
            self.dither_level = config['dither_level']


    def __call__(self, filename):
        """ This is called for each file """
        # remove dependency on libsndfile. Only accept wav, use Sox
        data, srate = audioread(filename, targetsr=self.sbpca.srate)
        assert srate == self.sbpca.srate
        # Actually run it
        ftrs =  self.sacc(data, srate)
        # Return the features
        return ftrs

    def sacc(self, data, srate):
        """
        Run the SAcC pitch tracker on the specified waveform/sampling rate
        using the configuration specified on construction
        Return two vectors, pitch (in Hz) and P(voicing) (posterior)
        """
        # Pad out d with zeros so get right number of winsamps frames
        # (and add unique dithering noise over whole signal)
        xdat = dithering(np.r_[data, np.zeros(self.sbpca.maxlags)],
                         self.dither_level)
        # Pre-allocate whole activations matrix
        nframes = self.sbpca.nframes(len(data))
#        acts = np.zeros( (len(self.net.obias), nframes) )
        acts = np.zeros( (len(self.net.obias), 0) )
#        (nChs, nDim, nLag) = np.shape(self.sbpca.mapping)
#        if self.output_pcas:
#            ftrs = np.zeros( (nChs, nDim, nframes) )
#        elif self.output_autocos:
#            ftrs = np.zeros( (nChs, nLag, nframes) )
#        else:
#            ftrs = np.zeros( (2, nframes) )

        framesamps = self.sbpca.framesamps
        # How many frames to process each time in loop
        #blockframes = 100
        blockframes = max(1, int(np.ceil(self.n_s * (srate/framesamps))))
        blocksamps = blockframes * framesamps
        nblocks = int(np.ceil(float(nframes) / float(blockframes)))

        # How many frames do we try to prepad?
        prepadframes = 10

        isfirst = 1
        donefr = 0
        for block in range(nblocks):
            # Figure next block of samples, including pre- and post-padding
            actualprepadframes = min(prepadframes, block*blockframes)
            blockbasesamp = block*blocksamps
            blocklastsamp = min(len(xdat), blockbasesamp + blocksamps 
                                +self.sbpca.padsamps)
            xpts = xdat[(blockbasesamp - actualprepadframes*framesamps)
                        :blocklastsamp]
            # Run the sbpca part
            acs = self.sbpca.calc_autocos(xpts, srate, isfirst)
            (nsb, nlg, nfr) = np.shape(acs) # 24, 200, 501
            # Now we know how many frames this block...
            nactfr = nfr - actualprepadframes
            ftr = np.zeros( (nactfr, 0), float)
            frixs = range(donefr, donefr+nactfr)
            donefr += nactfr
            if self.write_rownum:  
                #ftr = np.c_[ftr, np.array(frixs, ndmin=2).transpose()]
                ftr = np.c_[ftr, 
                            self.start_utt * np.ones( (nactfr, 1), float), 
                            np.array(frixs, ndmin=2).transpose()]
            if self.write_time:
                ftr = np.c_[ftr, self.sbpca.ac_hop 
                                  * np.array(frixs, ndmin=2).transpose()]
            #blockix = range(block*blockframes, block*blockframes+nactfr)
            if self.write_sbac:
                ftr = np.c_[ftr, np.reshape(ftr[:, :, actualprepadframes:], 
                                            (nsb*nlg, nactfr)).transpose()]
            pcas = sbpca.sbpca_pca(acs[:, :, actualprepadframes:], 
                                   self.sbpca.mapping)
            (nsb, npc, nactfr) = np.shape(pcas)
            #pcasr = np.reshape(pcas, (nsb*npc, nactfr)).transpose()
            # Required order of each frame is pcdim slowest, subband fastest!
            pcasr = pcas.transpose().reshape((nactfr, nsb*npc))
            if self.write_sbpca:
                ftr = np.c_[ftr, pcasr]
            # Run the MLP classifier
            act = self.net.apply(pcasr).transpose()
            #acts[:,blockix] = act
            acts = np.c_[acts, act]
            if self.write_posteriors:
                ftr = np.c_[ftr, acts]
            if isfirst:
                isfirst = 0
                ftrs = ftr
            else:
                ftrs = np.r_[ftrs, ftr]        

        if self.write_pitch:
            # Run viterbi decode on all activations stitched together
            pth = sbpca_viterbi(acts, self.hmm_vp)
            # Convert pitch bin indices to frequencies in Hz by table lookup
            ftrs = np.c_[ftrs, self.ptchtab[pth]]
            
        # first activation is Pr(unvoiced), so Pr(voiced) is its complement
        if self.write_pvx:
            ftrs = np.c_[ftrs, 1.0 - acts[0,]]
    	
        return ftrs


############## Provide a command-line wrapper

def main(argv):
    """ Main routine to calculate SAcC from wav file """
    if len(argv) != 3:
        raise NameError( ("Usage: ", argv[0], 
                          " inputsound.wav outputpitchtrack.txt") )

    inwavfile = argv[1]
    outptfile = argv[2]

    # Setup config
    config = {}
    # sbpca params
    # diff file for py
    config['pca_file']    = 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat' 
    #config['kdim'] = 10 # inferred from mapping file
    config['nchs']        = 24
    config['n_s']         = 5.0  # secs per process block, controls blockframes
    config['SBF_sr']      = 8000.0
    config['SBF_fmin']    = 100.0
    config['SBF_bpo']     = 6.0
    config['SBF_q']       = 8.0  # not actually used for SlanPat ERB filters
    config['SBF_order']   = 2  # not actually used for SlanPat ERB filters
    config['SBF_ftype']   = 2  # ignored - python is always SlanPat ERB
    config['twin']        = 0.025  # autoco window len
    thop = 0.010
    config['thop']        = thop  # autoco hop
    # mlp params
    #config['wgt_file']    = 'aux/rats_sr8k_bpo6_sb24_k10_aCH_h100.wgt'
    #config['norms_file']  = 'aux/tr_rats_sr8k_bpo6_sb24_k10.norms'
    config['wgt_file']    = 'aux/sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt'
    config['norms_file']  = 'aux/tr_keele_rbf_pinknoise_sr8000_bpo6_nchs24_k10.norms'
    #config['nhid'] = 100 # inferred from wgt file, + input size from norms file
    #config['nmlp'] = 68  # output layer size, inferred from wgt file
    config['pcf_file']    = 'aux/pitch_candidates_freqz.txt'
    # viterbi decode params
    config['hmm_vp']      = 0.9 # interpretation changed c/w Matlab
    # output options
    config['write_rownum'] = 0 # prepend row number
    config['write_time']  = 1  # prepend time in seconds to output
    config['write_sbac'] = 0   # output raw autocorrelations (big - 24 x 200)
    config['write_sbpca'] = 0  # output subband pcas (24 x 10)
    config['write_posteriors'] = 0 # output raw pitch posteriors (68)
    config['write_pitch'] = 1  # output the actual pitch value in Hz (1)
    config['write_pvx'] = 1    # output just 1-posterior(unvoiced) (1)
    # Tricks with segmenting utterances not implemented in Python
    config['start_utt'] = 0    # what utterance number to start at
    #config['incr_utt'] = 0     # increment the utterance each seg (?)
    #config['segs_per_utt'] = 1 # break each utterance into this many segs
    config['verbose'] = 0
    #config['disp'] = 0         # no display code in Python
    # Output file format is the concern of the calling layer
    #config['sph_out'] = 0
    #config['mat_out'] = 0
    #config['txt_out'] = 1
    config['dither_level'] = 1e-3

    # Configure
    sacc_extractor = SAcC(config)

    # Apply
    features = sacc_extractor(inwavfile)

    # Write the data out
    np.savetxt(outptfile, features, fmt='%.3f', delimiter=' ', newline='\n')


# Run the main function if called from the command line
if __name__ == "__main__":
    import sys
    main(sys.argv)

