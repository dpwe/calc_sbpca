########################
"""
# mlp.py
#
# functions for manipulating MLPs as written by QuickNet
# for sbpca 
#
# 2013-08-15 Dan Ellis dpwe@ee.columbia.edu
"""
import numpy as np
import os.path

############# from mlp_load.m


class MLP:
    """  Class to hold MLP definition """
    # Weights
    i_h = None  # np.zeros( (nhid, nin) )  # Input to Hidden
    h_o = None  # np.zeros( (nout, nhid) ) # Hidden to Output
    hbias = None  # np.zeros( nhid )         # Hidden layer Bias
    obias = None  # np.zeros( nout )         # Output layer Bias
    ofs = None # np.zeros( nin )          # feature normalization offset
    sca = None # np.ones( nin )           # feature normalization scaling

    def __init__(self, wtsfile, normsfile=None):
        # Setup the parameters from a wtsfile/normsfile pair

        if normsfile == None:
            # generate norms file name from wts file name
            head, tail = os.path.split(wtsfile)
            root, ext = os.path.splitext(tail)
            normsfile = head + root + '.norms'

        # Read the normsfile first, so we can figure out the input layer size
        fid = open(normsfile, 'r')
        # First is vector of the ofsets
        self.ofs, isize = readvec(fid)
        # Second is vector of the scale factors
        self.sca, isize = readvec(fid, isize)
        fid.close()

        # Now read the weights file
        fid = open(wtsfile, 'r')
        # first is IH weights
        self.i_h, hsize = readweigvec(fid, isize)
        # second is HO weights
        self.h_o, osize = readweigvec(fid, hsize)
        # then hidden layer bias
        self.hbias, hsize = readvec(fid, hsize, 'biasvec')
        # then output layer bias
        self.obias, osize = readvec(fid, osize, 'biasvec')
        # We're done
        fid.close()

    def apply(self, inp):
        """
        output = mlp.apply(input)
        apply the MLP to each row of the input array to create rows of 
        the output array.
        <input> - input data, <nitems> x <ninput>
        <output> - output results, <nitems> x <noutput>
        """
        #nfrms = np.size(inp, axis=0)
        # Normalize with offsets and scales.  Broadcasting does implicit repmat
        d_n = (inp - self.ofs)*self.sca
        # Apply first layer weights
        hlo = 1. / (1. + np.exp(-np.dot(d_n, self.i_h.transpose()) 
                                 - self.hbias))
        # Second layer
        ole = np.exp(np.dot(hlo, self.h_o.transpose()) + self.obias)
        # Apply softmax normalization
        return ((ole.transpose()/ole.sum(axis=1))).transpose()


def readvec(fid, ilen=None, keyword='vec'):
    """
    vec, olen = readvec(fid, ilen=None)
    Read a vector of values from a ascii (norms) file, return it and its size <olen>. 
    If provided, <ilen> is the expected size, raise an error if mismatched.
    """
    tokens = fid.readline().split()
    assert tokens[0] == keyword
    olen = int(tokens[1])
    if ilen != None:
        assert olen == ilen
    return ([float(fid.readline()) for i in range(olen)], 
            olen)


def readweigvec(fid, cols):
    """
    array, rows = readweigvec(fid, cols)
    Read a matrix of weight vectors from an open MLP weights file <fid>.
    Fastest dimension <cols> is passed in, <rows> is inferred from total length
    returned <array> is rows x cols
    """
    vals, totsize = readvec(fid, None, 'weigvec')
    rows = totsize / cols
    assert cols * int(rows) == totsize
    return  ( np.reshape(vals, (rows, cols) ), 
              rows )

