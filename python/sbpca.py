#
# sbpca - subband principal component analysis - core components
#         split out of SAcC.py
#
# 2013-09-19 Dan Ellis dpwe@ee.columbia.edu

import math
import numpy as np
import scipy.signal

# c extension to calculate autocorr
import _autocorr_py

#####################################
############# from sbpca_filterbank.m

# Class to hold the fbank returns
class fbank_def:
    a = None # np.zeros( (bands, falen), float);
    b = None # np.zeros( (bands, fblen), float);
    t = None  # np.zeros( bands, float);
    bw = None # np.zeros( bands, float);
    cf = None # np.zeros( bands, float);
    zi = None # np.zeros( (bands, max(falen,fblen)), float);
    sr = None # float

    def clear_zi(self):
        # clear initial state
        self.zi = np.zeros( (np.size(self.a, axis=0), 
                             max(np.size(self.b, axis=1),
                                 np.size(self.a, axis=1))-1) )

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

    fbank.sr = sr;

    fbank.clear_zi()

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
               *(np.cos(2*cf*math.pi*T) - math.sqrt(3.0 - 2.0**(3.0/2.0))
                 * np.sin(2*cf*math.pi*T))) 
             * (-2*np.exp(4j*cf*math.pi*T)*T 
                 + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
                 * (np.cos(2*cf*math.pi*T) + math.sqrt(3.0 - 2.0**(3.0/2.0)) 
                    * np.sin(2*cf*math.pi*T)))
             * (-2*np.exp(4j*cf*math.pi*T)*T 
                 + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
                 * (np.cos(2*cf*math.pi*T) 
                    - math.sqrt(3.0 + 2.0**(3.0/2.0))*np.sin(2*cf*math.pi*T)))
             * (-2*np.exp(4j*cf*math.pi*T)*T + 2*np.exp(-(B*T) + 2j*cf*math.pi*T)*T
                 * (np.cos(2*cf*math.pi*T) 
                    + math.sqrt(3.0 + 2.0**(3.0/2.0))*np.sin(2*cf*math.pi*T))) 
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

def sbpca_subbands(d,sr,fbank, discard=0, isfirst=0):
    """
    % subbands, freqs = sbpca_subbands(d,sr,fbank,discard)
    %   Filter into subbands for sbpca
    %   freqs returns the center frequencies for each subband.
    %   subbands is <nchs = 24 x ntime == length(d)>
    %   if discard is > 0, it means that the input signal was padded 
    %   at the end with this many frames, but their effect on the state 
    %   should not be recorded; instead, the next call to sbpca_subband 
    %   starts from the state after d[0:-discard]
    % 2013-05-27 Dan Ellis dpwe@ee.columbia.edu
    """

    # recover number of filters
    bands = len(fbank.b)

    # find size of d
    xsize = len(d)

    # initialize output array to full size
    # transpose domain  - avoids quite so much swapping during inner loop
    subbands = np.zeros( (bands,xsize) )

    # calculate each row
    for filt in range(bands):
        # disp(['band ' int2str(filt)]);
        # pad t zeros on the end, since we're going to chop from tail
        t = np.round(fbank.t[filt])
#        y = scipy.signal.lfilter(fbank.b[filt,], 
#                                 fbank.a[filt,], 
#                                 np.r_[d, np.zeros(t)])
        sig = np.r_[d, np.zeros(t)]
        # run and update state
        if discard > 0:
            y, fbank.zi[filt,] = scipy.signal.lfilter(fbank.b[filt,], 
                                                      fbank.a[filt,], 
                                                      sig[:-discard], 
                                                      zi=fbank.zi[filt,])
            # run last part without storing final state
            y2, zjunk = scipy.signal.lfilter(fbank.b[filt,], 
                                             fbank.a[filt,], 
                                             sig[-discard:], 
                                             zi=fbank.zi[filt,])
            y = np.r_[y, y2]
        else:
            y, fbank.zi[filt,] = scipy.signal.lfilter(fbank.b[filt,], 
                                                      fbank.a[filt,], 
                                                      sig, 
                                                      zi=fbank.zi[filt,:])           
        # shift the output to discard the first <t> samples
        y = y[t:]
        # HW rectify the signal
        subbands[filt,] = np.maximum(y,0)

    return subbands, fbank.cf

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
    (autocos, lags) = autocorrelogram(subbands, sr, maxlags, hop, win)
    
    #autocos = acf
    #print "no normalization"

    # Make it lags x subbands x timeframes
    #autocos = permute(autocos, [2 1 3])
    
    return autocos #, acf_engy

############## from autocorrelogram.m

def autocorrelogram(x, sr, maxlags=None, h=0.010, w=0.025):
    """
    function [ac, lags] = autocorrelogram(x, sr, maxlags, h, w)
    %    
    % x is a input signal consisting of multiple rows, each a separate
    % channel, and sr is samplingrate.  
    % Using w sec windows at every h-sec length of frame, 
    % calculate normalized autocorrelation ac and its energy sc of  the input signal.  
    %
    % kslee@ee.columbia.edu, 6/16/2005
    %
    """
    
    if maxlags == None:
        maxlags = int(np.round(sr * w))
    
    # Change time into the length of frame and window
    frmL = int(sr*h)
    winL = int(sr*w)
    
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

    #print "nchs=%d maxlags=%d nfrms=%d" % (nchs, maxlags, nfrms)
    
    #ac = np.zeros( (nchs, maxlags, nfrms) )
    #sc = np.zeros( (nchs, maxlags, nfrms) )
    autocos = np.zeros( (nchs, maxlags, nfrms) )

    for ch in range(nchs):
        xx = x[ch,]
        # This is the pure-python version
        #autocos[ch,] = my_autocorr(xx,frmL,nfrms,maxlags,winL)
        # This is the optimized C version, about 20% faster...
        autocos[ch,] = _autocorr_py.autocorr(xx, frmL, nfrms, maxlags, winL).transpose()

    #autocos = acf / (acf_engy + (acf_engy == 0))
    return autocos, lags


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################ from mitch's example
def frames_from_signal(signal, window, shift, mode=None):
    frames = rolling_window(signal.T, window)
    oframes = frames[..., 0::shift, :]
    return np.rollaxis(oframes, 1)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def my_autocorr(X,frmL,nfrms,maxlags,winL):
    """
    function c = my_autocorr(X,frmL,nfrms,maxlags,winL)
    % pure-matlab version of what autocorr.mex calculates
    % (excepting cumulated errors from single precision)
    """
    
    c = np.zeros( (maxlags, nfrms) )
    
    # Optimizations from Mitch McLaren, <mitchell.mclaren@sri.com>
    w2mat = frames_from_signal(X, winL+maxlags-1, frmL)
    w1mat = frames_from_signal(X, winL, frmL)[:,:nfrms]

    for f in range(nfrms):
        c[:,f] = np.correlate(w2mat[:,f], w1mat[:,f])

    w22mat = w2mat**2
    sc = np.cumsum(np.r_[w22mat, np.zeros((winL,nfrms))] 
                   - np.r_[np.zeros((winL,nfrms)), w22mat], axis=0)
    sc[sc<0] = 0 # per Mitch 2013-09-19
    s = np.sqrt(sc[winL-1,:]*sc[winL-1:-winL,:])

    return c/(s + (s==0))

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

class sbpca(object):
    """
    Object to encapsulate sbpca configuration parameters & execution
    """
    def __init__(self, config):
        """ Initialize default values """
        self.config = config
        self.sr = config['SBF_sr']
    	self.fbank = sbpca_filterbank(self.sr, config['SBF_fmin'], 
                                      config['SBF_bpo'], config['nchs'], 
                                      config['SBF_q'], 
                                      config['SBF_order'])
    	M = scipy.io.loadmat(config['pca_file'])
    	self.mapping = M["mapping"]
        self.maxlags = np.size(self.mapping, axis=2)
        self.ac_win = config['ac_win']
        self.ac_hop = config['ac_hop']
        self.framesamps = int(np.round(self.ac_hop * self.sr))
        self.winsamps = int(np.round(self.ac_win * self.sr))
        # extra to read in the end to allow last frame to be fully calculated
        # read by clients
        self.padsamps = self.winsamps - self.framesamps + self.maxlags + int(np.max(self.fbank.t))


    def nframes(self,lend):
        """
        How many frames does a sound of <lend> samples produce?
        """
        return int(max(0, 1 + np.floor((lend - self.winsamps)/self.framesamps)))

    def calc_autocos(self, d, sr, isfirst=0):
        """
        Calculate subband autocorrelation from audio.
        If "isfirst" is set, clear the filter state and warm up again
        """
        # Run the processing stages block by block
        if isfirst:
            self.fbank.clear_zi()
            npad = self.padsamps
        else:
            npad = 0
        sbs,frqs = sbpca_subbands(d, sr, self.fbank, npad, isfirst)
        acs = sbpca_autoco(sbs, sr, self.ac_win, self.ac_hop, 
                           self.maxlags)
        return acs

    def calc_sbpcas(self, d, sr, isfirst=0):
        """
        Calculate SBPCA for some features.
        Successive blocks will stick together, but it doesn't at the 
        moment handle partial blocks between frames.
        If "isfirst" is set, clear the filter state and warm up again
        """
        acs = self.calc_autocos(d, sr, isfirst)
        pcas     = sbpca_pca(acs, self.mapping)
        return pcas

