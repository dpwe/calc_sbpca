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
import scipy.cluster

import sbpca


# for main
import sys
# For SRI's wavreading code
import scipy.io.wavfile as wav


################### from spca_vqs.m

def vquantize(pcas, vqcodebook, vqmeans, vqstds):
    """
    % vqs = sbpca_vqs(pcas, params)
    %    Convert subband principal components to vector quantized values
    %    for SBPCA.
    %    pcas are (pcadim == 10, chans == 24, timefrm)
    %    vqs are (blocks == 4, timefrm)
    %    each value is one VQ index (0..1023).
    % 2013-08-03 Dan Ellis dpwe@ee.columbia.edu
    """

    (n_rec, n_ftr, k_dim) = np.shape(vqcodebook) # 4 x 1000 x 60
    (n_chs, n_dim, n_tim) = np.shape(pcas)    # 24 x 10 x T

    # Reshape pcas into vectors to be quantized
    grouping = int(n_chs / n_rec) # better hope it's 24 / 4 = 6

    # allocate
    vqs = np.zeros( (n_rec, n_tim), int)

    for i in range(n_rec):
        chs = range(i*grouping, (i+1)*grouping)
        # 60 dim codewords in Matlab file are ordered with the 6 chans as
        # the fastest rotating dimension, so make sure python array does this
        # too before collapsing 10x6 into rows of 60
        vqs[i, ] = ac_vq_quantize(pcas[chs, :, :].transpose(1, 0, 2)
                                .reshape(grouping*n_dim, n_tim).transpose(),
                                vqcodebook[i, ], vqmeans[i, ], vqstds[i, ])

    return vqs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ac_vq_quantize(data, vqcodes, vqmean, vqstd):
    """
    % I = acVQQuantize(D,CB)
    %    Return I as the quantization of each data row D according to
    %    the codebook CB.mean, CB.std, CB.codewords.
    % 2012-07-31 Dan Ellis dpwe@ee.columbia.edu
    """
    (n_tim, n_d) = np.shape(data)
    maxdrows = min(10000, n_tim)
    nblocks = int(math.ceil(n_tim/maxdrows))

    indx = np.zeros( n_tim, int)

    if n_d > 0:
        for blk in range(nblocks):
            bkix = range(blk*maxdrows, min(n_tim, (blk+1)*maxdrows))
            # Extract rows and apply mean/var normalization
            d_block = (data[bkix, ] - vqmean) / vqstd

            # Quantize
            indx[bkix] = scipy.cluster.vq.vq(d_block, vqcodes)

    return indx



#####################################
def hist(vqs, winframes, hopframes, codesperblock):
    """
    % hists = sbpca_hist(vqs, params)
    %    Collapse the vq codewords into per-segment histograms.
    %    Each column of hists is the mean of all the frames in
    %    params.histwin seconds every params.histhop seconds.
    %
    % 2013-08-11 Dan Ellis dpwe@ee.columbia.edu rebuild of sbcpa calculation
    """

    #winframes = int(np.round(params.histwin / params.hoptime))
    #hopframes = int(np.round(params.histhop / params.hoptime))

    codeblocks, vqslen = np.shape(vqs)

    nblocks = int(np.round(vqslen/float(hopframes)))

    #[codeblocks, ndim, codesperblock] = size(params.vqcodebook);
    #% 4, (60), 1000

    hists = np.zeros( (nblocks, codeblocks*codesperblock), float)

    for blk in range(nblocks):
        frameix = range(blk*hopframes, min(blk*hopframes+winframes, vqslen))
        for cbk in range(codeblocks):
            for frm in frameix:
                idx = cbk*codesperblock + vqs[cbk, frm]
                hists[blk, idx] += 1.0
        hists[blk, :] /= len(frameix) * codeblocks

    return hists





#####################################

# Main class

class AudFtr(object):
    """
    Compute sbpca-based auditory features
    meaning 4000-dim histograms over 2 sec windows.
    """
    def __init__(self, conf):
        """ Initialize default values """
        #self.config = config
        # initialize sbpca subsystem
        self.sbpca = sbpca.SbPca(conf)
        # set up AudFtr-specific params
        # Read in the VQ codebooks
        mat_cb = scipy.io.loadmat(conf['vq_file'])
        # Read the codebook as rects x codewords x dims (4 x 1000 x 60)
        self.vqcodebook = mat_cb["codebook"].transpose( (0, 2, 1) )
        self.vqmeans = mat_cb["recMean"]
        self.vqstds  = mat_cb["recStd"]
        self.hist_win = conf["hist_win"]
        self.hist_hop = conf["hist_hop"]

    def __call__(self, data, srate):
        """
        Run the sbpcahist ftr extractor on the specified waveform/sampling rate
        using the configuration specified on construction
        Return a matrix of <ftrs> x <blocks>
        """
        # Calculate the subband PCA features
        pcas     = self.sbpca.calc_sbpcas(data, srate)
        # Vector quantize in individual frames
        vqs      = vquantize(pcas, self.vqcodebook, self.vqmeans, self.vqstds)
        # Collapse blocks into histograms

        hists    = hist(vqs,
                        int(np.round(self.hist_win / self.sbpca.ac_hop)),
                        int(np.round(self.hist_hop / self.sbpca.ac_hop)),
                        np.shape(self.vqcodebook)[1])

        return hists


############## Provide a command-line wrapper

def main(argv):
    """ Main routine to calculate audftr files from command line """
    if len(argv) != 3:
        raise NameError( ("Usage: ", argv[0],
                          " inputsound.wav outputaudftr.txt") )

    inwavfile = argv[1]
    outptfile = argv[2]

    # Setup config
    config = {}
    # sbpca params
    # diff file for py
    config['pca_file']    = 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat'
    config['nchs']        = 24
    config['SBF_sr']      = 8000
    config['SBF_fmin']    = 100
    config['SBF_bpo']     = 6
    config['SBF_q']       = 8  # not actually used for SlanPat ERB filters
    config['SBF_order']   = 2  # not actually used for SlanPat ERB filters
    config['twin']        = 0.025  # autoco window len
    config['thop']        = 0.010  # autoco hop
    # audftr params
    config['vq_file']     = '../CB-sbpca-4x60x1000.mat'
    config['hist_win']    = 2.0    # histogram pooling window len
    config['hist_hop']    = 2.0    # histogram hop

    # Configure
    ftr_extractor = AudFtr(config)

    # Read in wav file
    srate, wavd = wav.read(inwavfile)
    # normalize short ints to floats of -1 / 1
    data = np.asfarray(wavd) / 32768.0

    # Apply
    features = ftr_extractor(data, srate)

    # Write the data out
    nfr = np.size(features, axis=0)
    np.savetxt(outptfile,
               np.c_[np.zeros(nfr), range(nfr), features],
               fmt='%.4f', delimiter=' ', newline='\n')

# Actually run main
if __name__ == "__main__":
    main(sys.argv)
