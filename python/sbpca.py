"""
# sbpca - subband principal component analysis - core components
#         split out of SAcC.py
#
# 2013-09-19 Dan Ellis dpwe@ee.columbia.edu
"""

import math
import numpy as np
import scipy.signal

# c extension to calculate autocorr
try:
  import _autocorr_py
  autoco_ext = True
except ImportError:
  autoco_ext = False

#####################################
############# from sbpca_filterbank.m

class FbankDef(object):
    """
    FbankDef
    Class to hold the fbank configuration variables
    """
    a_i = None # np.zeros( (bands, falen), float);
    b_i = None # np.zeros( (bands, fblen), float);
    tdel = None  # np.zeros( bands, float);
    bwdth = None # np.zeros( bands, float);
    cfreq = None # np.zeros( bands, float);
    z_i = None # np.zeros( (bands, max(falen,fblen)), float);
    srate = None # float

    def clear_zi(self):
        """
        # FbankDef.clear_zi() clear initial state fields given filter defs
        """
        self.z_i = np.zeros( (np.size(self.a_i, axis=0),
                             max(np.size(self.b_i, axis=1),
                                 np.size(self.a_i, axis=1))-1) )

    def __init__(self):
        self.a_i = None
        self.b_i = None

def filterbank(srate=8000.0, fmin=100.0, bpo=6.0, bands=24, q_f=8.0, order=2):
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

    fbank = FbankDef()

    fbank.cfreq = np.zeros( bands )

    for filt in range(bands):
        fbank.cfreq[filt] = math.exp(logminfreq + filt*logfreqfactor)

    fbank.b_i, fbank.a_i, fbank.tdel, fbank.bwdth = make_erb_filter(srate,
                                                                    fbank.cfreq)

    fbank.srate = srate

    fbank.clear_zi()

    return fbank

# subplot(111)
# for i = 1:size(fbank.b,1)
#   [hh,ww] = freqz(fbank.b(i,:),fbank.a(i,:));
#   plot(ww/pi, 20*log10(abs(hh)));
#   hold on;
# end
# hold off;

def make_erb_filter(fs, cf, cq=0):
    """
    % [B,A,Tds,BW] = make_erb_filter(fs,cf,CQ)     Design a Patterson cochlea filter.
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

    t = 1.0/fs

    # Change the following parameters if you wish to use a different ERB scale.
    ear_q = 9.26449  #  Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1.0

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear
    # Filter Bank."
    if cq == 0:
        erb = ((cf/ear_q)**order + min_bw**order)**(1/order)
    else:
        # True-CQ hack - for better phase alignment of filters
        erb = (cf/ear_q)*((fs/8)/ear_q+min_bw)*ear_q/(fs/8)

    bw = 1.019*2*math.pi*erb

    # Below here, just cf, t and bw used

    gain = abs((-2*np.exp(4j*cf*math.pi*t)*t + 2*np.exp(-(bw*t)
                                                         + 2j*cf*math.pi*t)*t
                 *(np.cos(2*cf*math.pi*t) - math.sqrt(3.0 - 2.0**(3.0/2.0))
                   * np.sin(2*cf*math.pi*t)))
               * (-2*np.exp(4j*cf*math.pi*t)*t
                   + 2*np.exp(-(bw*t) + 2j*cf*math.pi*t)*t
                   * (np.cos(2*cf*math.pi*t) + math.sqrt(3.0 - 2.0**(3.0/2.0))
                      * np.sin(2*cf*math.pi*t)))
               * (-2*np.exp(4j*cf*math.pi*t)*t
                   + 2*np.exp(-(bw*t) + 2j*cf*math.pi*t)*t
                   * (np.cos(2*cf*math.pi*t)
                      - math.sqrt(3.0 + 2.0**(3.0/2.0))*np.sin(2*cf*math.pi*t)))
               * (-2*np.exp(4j*cf*math.pi*t)*t + 2*np.exp(-(bw*t)
                                                           + 2j*cf*math.pi*t)*t
                   * (np.cos(2*cf*math.pi*t)
                      + math.sqrt(3.0+2.0**(3.0/2.0))*np.sin(2*cf*math.pi*t)))
               / (-2 / np.exp(2*bw*t) - 2*np.exp(4j*cf*math.pi*t)
                   + 2*(1 + np.exp(4j*cf*math.pi*t))/np.exp(bw*t))**4)
    ncf = len(cf)
    feedback = np.zeros((ncf, 9))
    forward = np.zeros((ncf, 5))
    forward[:, 0] = t**4 / gain
    forward[:, 1] = -4*t**4*np.cos(2*cf*math.pi*t)/np.exp(bw*t)/gain
    forward[:, 2] = 6*t**4*np.cos(4*cf*math.pi*t)/np.exp(2*bw*t)/gain
    forward[:, 3] = -4*t**4*np.cos(6*cf*math.pi*t)/np.exp(3*bw*t)/gain
    forward[:, 4] = t**4*np.cos(8*cf*math.pi*t)/np.exp(4*bw*t)/gain
    feedback[:, 0] = np.ones(ncf)
    feedback[:, 1] = -8*np.cos(2*cf*math.pi*t)/np.exp(bw*t)
    feedback[:, 2] = 4*(4 + 3*np.cos(4*cf*math.pi*t))/np.exp(2*bw*t)
    feedback[:, 3] = -8*(6*np.cos(2*cf*math.pi*t)
                         + np.cos(6*cf*math.pi*t))/np.exp(3*bw*t)
    feedback[:, 4] = 2*(18 + 16*np.cos(4*cf*math.pi*t)
                        + np.cos(8*cf*math.pi*t))/np.exp(4*bw*t)
    feedback[:, 5] = -8*(6*np.cos(2*cf*math.pi*t)
                         + np.cos(6*cf*math.pi*t))/np.exp(5*bw*t)
    feedback[:, 6] = 4*(4 + 3*np.cos(4*cf*math.pi*t))/np.exp(6*bw*t)
    feedback[:, 7] = -8*np.cos(2*cf*math.pi*t)/np.exp(7*bw*t)
    feedback[:, 8] = np.exp(-8*bw*t)

    # from differentiating the envelope function, t**(n-1)np.exp(-t/wb)
    n = 4
    t_ds = fs*(n-1)/bw
    bandwidth = erb

    return (forward, feedback, t_ds, bandwidth)

############### from sbpca_subbands.m

def subbands(data, srate, fbank, discard=0, isfirst=0):
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

    if srate != fbank.srate:
        raise ValueError('sample rate mismatch')

    # recover number of filters
    bands = len(fbank.b_i)

    # find size of d
    xsize = len(data)

    # initialize output array to full size
    # transpose domain  - avoids quite so much swapping during inner loop
    subbands = np.zeros( (bands, xsize) )

    # calculate each row
    for filt in range(bands):
        # disp(['band ' int2str(filt)]);
        # pad t zeros on the end, since we're going to chop from tail
        tdel = np.round(fbank.tdel[filt])
#        y_1 = scipy.signal.lfilter(fbank.b_i[filt,],
#                                 fbank.a_i[filt,],
#                                 np.r_[data, np.zeros(tdel)])
        sig = np.r_[data, np.zeros(tdel)]
        # run and update state
        if discard > 0:
            y_1, fbank.z_i[filt,] = scipy.signal.lfilter(fbank.b_i[filt,],
                                                         fbank.a_i[filt,],
                                                         sig[:-discard],
                                                         zi=fbank.z_i[filt,])
            # run last part without storing final state
            y_2, zjunk = scipy.signal.lfilter(fbank.b_i[filt,],
                                              fbank.a_i[filt,],
                                              sig[-discard:],
                                              zi=fbank.z_i[filt,])
            y_1 = np.r_[y_1, y_2]
        else:
            y_1, fbank.z_i[filt,] = scipy.signal.lfilter(fbank.b_i[filt,],
                                                        fbank.a_i[filt,],
                                                        sig,
                                                        zi=fbank.z_i[filt,:])
        # shift the output to discard the first <t> samples
        y_1 = y_1[tdel:]
        # HW rectify the signal
        subbands[filt, ] = np.maximum(y_1, 0)

    return subbands, fbank.cfreq

############## from sbpca_autoco.m

def autoco(subbands, srate=8000, win=0.025, hop=0.010, maxlags=None):
    """
    % [autocos,energy] = sbpca_autocos(subbands,sr)
    %   Calculate autocorrelations over subbands for sbpca.
    %   subbands is nchs x ntime
    %   autocos  is nlag x nchs x nframes
    % 2013-05-27 Dan Ellis dpwe@ee.columbia.edu
    """
    if maxlags == None:
        maxlags = int(np.round(win*srate))

    # multichannel autocorrelation
    autocos, acf_energy = autocorrelogram(subbands, srate, maxlags, hop, win)

    #autocos = acf
    #print "no normalization"

    # Make it lags x subbands x timeframes
    #autocos = permute(autocos, [2 1 3])

    return autocos, acf_energy

############## from autocorrelogram.m

def autocorrelogram(data, srate, maxlags=None, hop=0.010, win=0.025):
    """
    function ac = autocorrelogram(x, sr, maxlags, h, w)
    %
    % x is a input signal consisting of multiple rows, each a separate
    % channel, and sr is samplingrate.
    % Using w sec windows at every h-sec length of frame,
    % calculate normalized autocorrelation ac and its energy sc of
    % the input signal.
    %
    % kslee@ee.columbia.edu, 6/16/2005
    %
    """

    if maxlags == None:
        maxlags = int(np.round(srate * win))

    # Change time into the length of frame and window
    frm_len = int(srate*hop)
    win_len = int(srate*win)

    # the channels and points of input signal
    nchs, npts = np.shape(data)

    # the total number of frames within each segment
    nfrms = int(1 + math.floor((npts + 1 - win_len - maxlags)/frm_len))
    # line 61 of autocorr.c is
    # 		z1 += xp[base+win_len+j]*xp[base+win_len+j+eta];
    # where base = (f-1)*frm_len;
    # and f = 0..(nfrms-1)
    # and j = 0..(win_len-1)
    # and eta = 0..(maxlags-1)
    #  and the index must stay within npts

    #print "nchs=%d maxlags=%d nfrms=%d" % (nchs, maxlags, nfrms)

    #ac = np.zeros( (nchs, maxlags, nfrms) )
    #sc = np.zeros( (nchs, maxlags, nfrms) )
    autocos = np.zeros( (nchs, maxlags, nfrms) )
    etot = np.zeros( (nchs, nfrms) )

    for chan in range(nchs):
        xdat = data[chan,]
        if autoco_ext:
            # This is the optimized C version.
            autocos[chan,] = _autocorr_py.autocorr(xdat, frm_len,
                                                   nfrms, maxlags, win_len).T
        else:
            # This is the pure-python version, about 20% slower.
            autocos[chan,], etotchan = my_autocorr(
                xdat, frm_len, nfrms, maxlags, win_len)
            etot[chan, ] = etotchan[chan, 0, ]

    #autocos = acf / (acf_engy + (acf_engy == 0))
    return autocos, etot


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

################ from mitch's example
def frames_from_signal(signal, window, shift):
    """
    frames_from_signal - convert a signal into a matrix of overlapped frames
    """
    frames = rolling_window(signal.T, window)
    oframes = frames[..., 0::shift, :]
    return np.rollaxis(oframes, 1)

def rolling_window(data, window):
    """
    rolling_window - extract successive instances of a window on a vector
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def my_autocorr(sig, frm_len, nfrms, maxlags, win_len):
    """
    function c = my_autocorr(sig,frm_len,nfrms,maxlags,win_len)
    % pure-matlab version of what autocorr.mex calculates
    % (excepting cumulated errors from single precision)
    """

    cor = np.zeros( (maxlags, nfrms) )

    # Optimizations from Mitch McLaren, <mitchell.mclaren@sri.com>
    w2mat = frames_from_signal(sig, win_len+maxlags-1, frm_len)
    w1mat = frames_from_signal(sig, win_len, frm_len)[:, :nfrms]

    for frm in range(nfrms):
        cor[:, frm] = np.correlate(w2mat[:, frm], w1mat[:, frm])

    w22mat = w2mat**2
    sums = np.cumsum(np.r_[w22mat, np.zeros((win_len, nfrms))]
                   - np.r_[np.zeros((win_len, nfrms)), w22mat], axis=0)
    sums[sums < 0] = 0 # per Mitch 2013-09-19
    etot = np.sqrt(sums[win_len-1, :]*sums[win_len-1:-win_len, :])

    return cor/(etot + (etot==0)), etot

############### from sbpca_pca.m
def pca(autocos, mapping):
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

    pcas = np.zeros( (nchs, pdims, nfrms) )
    for chan in range(nchs):
        #tmpac = autocos[chan, ]
        # To ignore frames that contain NaN elements
        #sumtmpac = np.sum(tmpac, axis=1);
        #tmpac[:, nonzero(sumtmpac==nan)] = 0;
        pcas[chan,] = np.dot(mapping[chan, ], autocos[chan, ])

    return pcas

class SbPca(object):
    """
    Object to encapsulate sbpca configuration parameters & execution
    """
    def __init__(self, config):
        """ Initialize default values """
        self.config = config
        self.srate = config['SBF_sr']
        self.fbank = filterbank(self.srate, config['SBF_fmin'],
                                config['SBF_bpo'], config['nchs'],
                                config['SBF_q'],
                                config['SBF_order'])
        matd = scipy.io.loadmat(config['pca_file'])
        self.mapping = matd["mapping"]
        self.maxlags = np.size(self.mapping, axis=2)
        self.ac_win = config['twin']
        self.ac_hop = config['thop']
        self.nchs = np.size(self.mapping, axis=0) # for clients to read
        self.framesamps = int(np.round(self.ac_hop * self.srate))
        self.winsamps = int(np.round(self.ac_win * self.srate))
        # extra to read in the end to allow last frame to be fully calculated
        # read by clients
        self.padsamps = self.winsamps - self.framesamps + self.maxlags
        # + int(np.max(self.fbank.t))

    def nframes(self, lend):
        """
        How many frames does a sound of <lend> samples produce?
        """
        return int(max(0, 1 + np.floor((lend - self.winsamps)/self.framesamps)))

    def calc_autocos(self, data, srate, isfirst=0):
        """
        Calculate subband autocorrelation from audio.
        If "isfirst" is set, clear the filter state and warm up again
        """
        # Run the processing stages block by block
        #print "calc_autocos: len(d)=", len(d), " isfirst=", isfirst
        if isfirst:
            self.fbank.clear_zi()
            npad = self.padsamps
        else:
            npad = 0
        sbs, frqs = subbands(data, srate, self.fbank, npad, isfirst)
        acs, etot = autoco(sbs, srate, self.ac_win, self.ac_hop,
                           self.maxlags)
        return acs

    def calc_sbpcas(self, data, srate, isfirst=0):
        """
        Calculate SBPCA for some features.
        Successive blocks will stick together, but it doesn't at the
        moment handle partial blocks between frames.
        If "isfirst" is set, clear the filter state and warm up again
        """
        acs = self.calc_autocos(data, srate, isfirst)
        pcas     = pca(acs, self.mapping)
        return pcas
