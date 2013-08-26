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

import mlp

#####################################
############# from sbpca_filterbank.m

# Class to hold the fbank returns
class fbank_def:
    a = None # np.zeros( (bands, falen), float);
    b = None # np.zeros( (bands, fblen), float);
    t = None  # np.zeros( bands, float);
    bw = None # np.zeros( bands, float);
    cf = None # np.zeros( bands, float);

def sbpca_filterbank(sr=8000.0, fmin=100.0, bpo=6.0, bands=24, q=8.0, ord=2):
    """
    Define the filterbank for sbpca.
    Return a structure with fields a, b, t, bw, cf
    each a vector with each element describing one of the filters
    """
    fmax = fmin*math.exp(math.log(2.0)*bands/bpo)
    
    if (fmin <= 0):
        raise RangeError('bpfiltbank: must be 0 < FMIN < FMAX (log scaling)')
    
    logfreqfactor = math.log(fmax/fmin)/bands
    logminfreq    = math.log(fmin)
    fmax = math.exp(logminfreq + bands * logfreqfactor)
    
    fbank = fbank_def()

    fbank.cf = np.zeros( bands )
    
    for filter in range(bands):
        fbank.cf[filter] = math.exp(logminfreq + filter*logfreqfactor)

    (fbank.b, fbank.a, fbank.t, fbank.bw) = MakeERBFilter(sr, fbank.cf)
     
    return fbank

# subplot(111)
# for i = 1:size(fbank.b,1)
#   [hh,ww] = freqz(fbank.b(i,:),fbank.a(i,:));
#   plot(ww/pi, 20*log10(abs(hh)));
#   hold on;
# end
# hold off;

def MakeERBFilter(fs,cf, cq=0):
  """
  % [B,A,Tds,BW] = MakeERBFilter(fs,cf,CQ)     Design a Patterson cochlea filter.
  % 	Computes the filter coefficients for a single Gammatone filter.
  %	The filters were defined by Patterson and Holdworth for simulating 
  % 	the cochlea.  The results are returned as arrays of filter
  % 	coefficients.  Each row of the filter arrays (forward and feedback)
  % 	can be passed to the MatLab "filter" function, or you can do all
  % 	the filtering at once with the ERBFilterBank() function.
  %	Tds is a vector of group delays in samples for each filter.
  %	BW is the bandwidth (in Hz) for each filter.
  % 	The filter bank contains filters for each frequency in <cf>
  %	If CQ is present and nonzero, force the filters to remain 
  %	const Q right down into the LF.
  % dpwe, after malcolm 1994sep19
  see https://engineering.purdue.edu/~malcolm/interval/1998-010/
  """
  # works for vector of cf, but handle a scalar too
  if type(cf) == float:
      cf = np.array([cf])

  T=1.0/fs
  # Change the following parameters if you wish to use a different ERB scale.
  EarQ = 9.26449  #  Glasberg and Moore Parameters
  minBW = 24.7
  order = 1.0
  
  # All of the following expressions are derived in Apple TR #35, "An
  # Efficient Implementation of the Patterson-Holdsworth Cochlear
  # Filter Bank."
  if cq==0:
    ERB = ((cf/EarQ)**order + minBW**order)**(1/order)
  else:
    # True-CQ hack - for better phase alignment of filters
    ERB = (cf/EarQ)*((fs/8)/EarQ+minBW)*EarQ/(fs/8)

  B=1.019*2*math.pi*ERB
  
  # Below here, just cf, T and B used
  
  gain = abs((-2*np.exp(4j*cf*math.pi*T)*T + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
               *(np.cos(2*cf*math.pi*T) - math.sqrt(3 - 2**(3/2))
                 * np.sin(2*cf*math.pi*T))) 
             * (-2*np.exp(4j*cf*math.pi*T)*T 
                 + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
                 * (np.cos(2*cf*math.pi*T) + math.sqrt(3 - 2**(3/2)) 
                    * np.sin(2*cf*math.pi*T)))
             * (-2*np.exp(4j*cf*math.pi*T)*T 
                 + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
                 * (np.cos(2*cf*math.pi*T) 
                    - math.sqrt(3 + 2**(3/2))*np.sin(2*cf*math.pi*T)))
             * (-2*np.exp(4j*cf*math.pi*T)*T + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
                 * (np.cos(2*cf*math.pi*T) 
                    + math.sqrt(3 + 2**(3/2))*np.sin(2*cf*math.pi*T))) 
             / (-2 / np.exp(2*B*T) - 2*np.exp(4j*cf*math.pi*T) 
                 + 2*(1 + np.exp(4j*cf*math.pi*T))/np.exp(B*T))**4)
  ncf = len(cf)
  feedback=np.zeros((ncf,9))
  forward=np.zeros((ncf,5))
  forward[:,0] = T**4 / gain
  forward[:,1] = -4*T**4*np.cos(2*cf*math.pi*T)/np.exp(B*T)/gain
  forward[:,2] = 6*T**4*np.cos(4*cf*math.pi*T)/np.exp(2*B*T)/gain
  forward[:,3] = -4*T**4*np.cos(6*cf*math.pi*T)/np.exp(3*B*T)/gain
  forward[:,4] = T**4*np.cos(8*cf*math.pi*T)/np.exp(4*B*T)/gain
  feedback[:,0] = np.ones(ncf)
  feedback[:,1] = -8*np.cos(2*cf*math.pi*T)/np.exp(B*T)
  feedback[:,2] = 4*(4 + 3*np.cos(4*cf*math.pi*T))/np.exp(2*B*T)
  feedback[:,3] = -8*(6*np.cos(2*cf*math.pi*T) + np.cos(6*cf*math.pi*T))/np.exp(3*B*T)
  feedback[:,4] = 2*(18 + 16*np.cos(4*cf*math.pi*T) + np.cos(8*cf*math.pi*T))/np.exp(4*B*T)
  feedback[:,5] = -8*(6*np.cos(2*cf*math.pi*T) + np.cos(6*cf*math.pi*T))/np.exp(5*B*T)
  feedback[:,6] = 4*(4 + 3*np.cos(4*cf*math.pi*T))/np.exp(6*B*T)
  feedback[:,7] = -8*np.cos(2*cf*math.pi*T)/np.exp(7*B*T)
  feedback[:,8] = np.exp(-8*B*T)
  
  # from differentiating the envelope function, t**(n-1)np.exp(-t/wb)
  n = 4
  Tds = fs*(n-1)/B
  BW = ERB
  
  return (forward,feedback,Tds,BW)

############### from sbpca_subbands.m

def sbpca_subbands(d,sr,fbank):
    """
    % [subbands,freqs] = sbpca_subbands(d,sr,params)
    %   Filter into subbands for sbpca
    %   freqs returns the center frequencies for each subband.
    %   subbands is <nchs = 24 x ntime == length(d)>
    % 2013-05-27 Dan Ellis dpwe@ee.columbia.edu
    """

    postpad = 0.04 # as in sub_cal_ac

    X = np.r_[dithering(d), np.zeros(np.round(postpad*sr))]

    # recover number of filters
    bands = len(fbank.b)

    # find size of X
    xsize = len(X)

    # initialize output array to full size
    # transpose domain  - avoids quite so much swapping during inner loop
    subbands = np.zeros( (bands,xsize) )

    # calculate each row
    for filt in range(bands):
        # disp(['band ' int2str(filt)]);
        # pad t zeros on the end, since we're going to chop from tail
        t = np.round(fbank.t[filt])
        y = scipy.signal.lfilter(fbank.b[filt,], 
                                 fbank.a[filt,], 
                                 np.r_[X, np.zeros(t)])
        # shift the output to discard the first <t> samples
        y = y[t:]
        # HW rectify the signal
        subbands[filt,] = np.maximum(y,0)

    return subbands, fbank.cf


def dithering(x):
    """
    % y = dithering(x)
    %    Add low-level noise to x to avoid digital zeros
    """
    # Ensure consistent random sequence
    np.random.seed(0)
    # Generate the dither sequence
    xlen = len(x)
    dither = np.random.rand(xlen) + np.random.rand(xlen) - 1
    # add it on 120 dB below the signal
    spow = np.std(x)
    return x + 1e-6 * spow * dither

############## from sbpca_autoco.m

def sbpca_autoco(subbands, sr=8000, win=0.025, hop=0.010, maxlags=None):
    """
    % [autocos,energy] = sbpca_autocos(subbands,sr)
    %   Calculate autocorrelations over subbands for sbpca.
    %   subbands is nchs x ntime
    %   autocos  is nlag x nchs x nframes
    % 2013-05-27 Dan Ellis dpwe@ee.columbia.edu
    """
    if maxlags == None:
        maxlags = int(np.round(win*sr))
    
    # multichannel autocorrelation
    (acf, acf_engy, lags) = autocorrelogram(subbands, sr, maxlags, hop, win)
    
    autocos = acf / (acf_engy + (acf_engy == 0))
    
    # Make it lags x subbands x timeframes
    #autocos = permute(autocos, [2 1 3])
    
    return autocos, acf_engy

############## from autocorrelogram.m

def autocorrelogram(x, sr, maxlags=None, h=0.010, w=0.025):
    """
    function [ac, sc, lags] = autocorrelogram(x, sr, maxlags, h, w)
    %    
    % x is a input signal consisting of multiple rows, each a separate
    % channel, and sr is samplingrate.  
    % Using w sec windows at every h-sec length of frame, 
    % calculate autocorrelation ac and its energy sc of  the input signal.  
    %
    % kslee@ee.columbia.edu, 6/16/2005
    %
    """
    
    if maxlags == None:
        maxlags = int(np.round(sr * w))
    
    # Change time into the length of frame and window
    frmL = sr*h
    winL = sr*w
    
    lags = range(maxlags)
    
    # the channels and points of input signal
    nchs, npts = np.shape(x)
    
    # the total number of frames within each segment
    nfrms = int(1 + math.floor((npts + 1 - winL - maxlags)/frmL))
    # line 61 of autocorr.c is 
    # 		z1 += xp[base+winL+j]*xp[base+winL+j+eta]; 
    # where base = (f-1)*frmL;
    # and f = 0..(nfrms-1)
    # and j = 0..(winL-1)
    # and eta = 0..(maxlags-1)
    #  and the index must stay within npts
    
    ac = np.zeros( (nchs, maxlags, nfrms) )
    sc = np.zeros( (nchs, maxlags, nfrms) )
    
    for ch in range(nchs):
        xx = x[ch,]
        (ac[ch,], sc[ch,]) = my_autocorr(xx,frmL,nfrms,maxlags,winL)

    return ac, sc, lags


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def my_autocorr(X,frmL,nfrms,maxlags,winL):
    """
    function [c,s] = my_autocorr(X,frmL,nfrms,maxlags,winL)
    % pure-matlab version of what autocorr.mex calculates
    % (excepting cumulated errors from single precision)
    """
    
    c = np.zeros( (maxlags, nfrms) )
    s = np.zeros( (maxlags, nfrms) )
    
    for f in range(nfrms):
        base = f*frmL
        w1 = X[ base:base+winL ]
        w2 = X[ base:base+winL+maxlags-1 ]
        #ac = np.correlate(w1, w2, "full")
        #c[:,f] = ac[ winL+maxlags-1 : winL-1 : -1 ]

        c[:,f] = np.correlate(w2, w1)

        sc = ( np.cumsum(np.r_[w2**2, np.zeros(winL)]) 
               - np.cumsum(np.r_[np.zeros(winL), w2**2]) )
        s[:,f] = np.sqrt(sc[winL-1]*sc[winL-1:winL-1+maxlags])

        w22 = w2**2
        sc = np.cumsum(np.r_[w22, np.zeros(winL)] 
                       - np.r_[np.zeros(winL), w22])
        s[:,f] = np.sqrt(sc[winL-1]*sc[winL-1:-winL])
    
    return c, s

############### from sbpca_pca.m
def sbpca_pca(autocos, mapping):
    """
    % pcas = sbpca_pca(autocos, params)
    %    Convert subband autocorrelations to principal components
    %    for SBPCA.
    %    autocos  is nchs x nlag x nframes
    %    mapping is the nchs x ndims x nlag PCA mapping matrix
    %    pcas is nchs x ndims x nfrms
    % 2013-05-27 Dan Ellis dpwe@ee.columbia.edu
    You load mapping with:
    M = scipy.io.loadmat('mapping-pca_sr8k_bpo6_sb24_k10.mat')
    mapping = M["mapping"]
    """

    (nchs, nlags, nfrms) = np.shape(autocos)

    (pchs, pdims, plags) = np.shape(mapping)

    pcas = np.zeros( (nchs,pdims,nfrms) )
    for c in range(nchs):
        #tmpac = autocos[c,]
        # To ignore frames that contain NaN elements
        #sumtmpac = np.sum(tmpac,axis=1);
        #tmpac[:,nonzero(sumtmpac==nan)] = 0;
        pcas[c,] = np.dot(mapping[c,], autocos[c,])
        
    return pcas

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

    # now calculate forward
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
        self.sr = config['SBF_sr']
    	self.fbank = sbpca_filterbank(self.sr, config['SBF_fmin'], 
                                      config['SBF_bpo'], config['nchs'], 
                                      config['SBF_q'], config['SBF_order'])
    	M = scipy.io.loadmat(config['pca_file'])
    	self.mapping = M["mapping"]
        self.maxlags = np.size(self.mapping, axis=2)
    	self.net = mlp.mlp(config['wgt_file'], config['norms_file'])
    	self.ptchtab = np.r_[0, np.loadtxt(config['pcf_file'])]
    	self.hmm_vp = config['hmm_vp']
        self.ac_win = config['ac_win']
        self.ac_hop = config['ac_hop']

    def __call__(self, filename):
        """ This is called for each file """
        sr, w = wav.read(filename)  # remove dependency on libsndfile. Only accept wav, use Sox
        d = np.asfarray(w) / 32768.0  # normalize short ints to floats of -1 / 1
        if sr == 16000 and self.sr == 8000:
            # slight trim to ss.decimate to make its phase align to matlab's resample
            d = ss.decimate(np.r_[d[1:],0], 2, ftype='fir')
            delay = 7
            d = np.r_[d[delay:],np.zeros(delay)]
            sr = sr/2
        assert sr == self.sr
        # Actually run it
        pitches, pvx =  self.SAcC(d, sr)
        # Return the features in one array
        return np.c_[pitches, pvx]

    def SAcC(self, d, sr):
        """
        Run the SAcC pitch tracker on the specified waveform/sampling rate
        using the configuration specified on construction
        Return two vectors, pitch (in Hz) and P(voicing) (posterior)
        """
        # Run the processing stages
    	sbs,frqs = sbpca_subbands(d, sr, self.fbank)
    	acs, ace = sbpca_autoco(sbs, sr, self.ac_win, self.ac_hop, self.maxlags)
    	pcas     = sbpca_pca(acs, self.mapping)
    	(nsb,npc,nfr) = np.shape(pcas)
    	acts     = self.net.apply(pcas.transpose().reshape(nfr, nsb*npc)).transpose()
    	pth      = sbpca_viterbi(acts, self.hmm_vp)

        # first activation is Pr(unvoiced), so Pr(voiced) is its complement
    	pvx = 1.0 - acts[0,]
    	# Convert pitch bin indices to frequencies in Hz by table lookup
    	pitches = self.ptchtab[pth]
        # Return one row of two values per frame
    	return pitches, pvx


############## Provide a command-line wrapper

if __name__=="__main__":

    import sys

    if len(sys.argv) != 3:
        raise NameError( ("Usage: ", sys.argv[0], " inputsound.wav outputpitchtrack.txt") )

    inwavfile = sys.argv[1]
    outptfile = sys.argv[2]

    # Setup config
    config = {}
    config['pca_file']    = 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat' # diff file for py
    #config['wgt_file']    = 'aux/rats_sr8k_bpo6_sb24_k10_aCH_h100.wgt'
    #config['norms_file']  = 'aux/tr_rats_sr8k_bpo6_sb24_k10.norms'
    config['wgt_file']    = 'aux/sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt'
    config['norms_file']  = 'aux/tr_keele_rbf_pinknoise_sr8000_bpo6_nchs24_k10.norms'
    config['pcf_file']    = 'aux/pitch_candidates_freqz.txt'
    config['nchs']        = 24
    config['SBF_sr']      = 8000
    config['SBF_fmin']    = 100
    config['SBF_bpo']     = 6
    config['SBF_q']       = 8  # not actually used for SlanPat ERB filters
    config['SBF_order']   = 2  # not actually used for SlanPat ERB filters
    config['ac_win']      = 0.025  # New option in Python - autoco window len
    config['ac_hop']      = 0.010  # New option in Python - autoco hop
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
            opf.write('%d %d %f %f\n' % (utt, i, features[i,0], features[i,1]))

