#!/usr/bin/env python2.7
#
# SAcC_list.py
#
# Direct replacement for the compiled Matlab version of SAcC
# that takes a list of file pairs as input.
#
# 2013-12-03 Dan Ellis dpwe@ee.columbia.edu

import SAcC
import srs
import re
import numpy as np

# default config
config = {
    'pca_file': 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat', 
    #'kdim': 10, # inferred from mapping file
    'nchs': 24,
    'n_s': 5.0,  # secs per process block, controls blockframes
    'SBF_sr': 8000.0,
    'SBF_fmin': 100.0,
    'SBF_bpo': 6.0,
    'SBF_q': 8.0,  # not actually used for SlanPat ERB filters
    'SBF_order': 2,  # not actually used for SlanPat ERB filters
    'SBF_ftype': 2,  # ignored - python is always SlanPat ERB
    'twin': 0.025,  # autoco window len
    'thop': 0.010, # autoco hop
    # mlp params
    'wgt_file': 'aux/sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt',
    'norms_file': 'aux/tr_keele_rbf_pinknoise_sr8000_bpo6_nchs24_k10.norms',
    #'nhid': 100, # inferred from wgt file, + input size from norms file
    #'nmlp': 68,  # output layer size, inferred from wgt file
    'pcf_file': 'aux/pitch_candidates_freqz.txt',
    # viterbi decode params
    'hmm_vp': 0.9, # interpretation changed c/w Matlab
    # output options
    'write_rownum': 1, # prepend row number
    'write_time': 0,  # prepend time in seconds to output
    'write_sbac': 0,   # output raw autocorrelations (big - 24 x 200)
    'write_sbpca': 0,  # output subband pcas (24 x 10)
    'write_posteriors': 0, # output raw pitch posteriors (68)
    'write_pitch': 1,  # output the actual pitch value in Hz (1)
    'write_pvx': 1,    # output just 1-posterior(unvoiced) (1)
    # Tricks with segmenting utterances
    'start_utt': 0,    # what utterance number to start at
    'incr_utt': 0 ,    # increment the utterance each seg (?)
    #'segs_per_utt': 1, # train_mkLabFile break utterances into this many segs
    #'disp': 0,         # no display code in Python
    # Output file format is the concern of the calling layer
    'sph_out': 0,
    'mat_out': 0,
    'txt_out': 1,
    # added 2014-04-10 after problems with Babel
    'dither_level': 1e-3, # noise is -60dB re: signal (was -120 dB, 1e-6)
    'verbose': 0
}

def my_savetext(filename, features):
    """ write feature file """
    #np.savetxt(filename, features, 
    #           fmt='%.3f', delimiter=' ', newline='\n')
    (rows, cols) = np.shape(features)
    with open(filename, 'w') as fp:
        for row in range(rows):
            fp.write('%d %d' % ( int(features[row, 0]), int(features[row, 1])))
            for col in range(cols - 2):
                fp.write(' %.3f' % features[row, col+2])
            fp.write('\n')

def sacc_write(outptfile, features, config):
    """ Write the output file for SAcC in format per config """
    if config['txt_out'] != 0:
        my_savetext(outptfile, features)
    elif config['mat_out'] != 0:
        raise NotImplementedError("MATLAB output not yet implemented")        
    elif config['sph_out'] != 0:
        raise NotImplementedError("SPHERE output not yet implemented")        
    else:
        raise SyntaxError("No valid output file type configuration")


def main(argv):
    """ Main routine to calculate SAcC from wav file """
    if len(argv) != 2 and len(argv) != 3:
        raise NameError( ("Usage: ", argv[0], 
                          " pairslist.txt [configfile.txt]") )

    pairslist = argv[1]

    if len(argv) == 3:
        # We have a config file
        configfile = argv[2]
        c = srs.Config(configfile)
        # Copy any defined fields into our config structure
        for var in c._get_varlist(None):
            val = c.vars[var]
            if re.match('[-0-9][0-9]*$', val) != None:
                val = int(val)
            elif re.match('[-0-9.][0-9.]*$', val) != None:
                val = float(val)
            config[var] = val

    with open(pairslist, 'r') as pairsfile:
        for line in pairsfile:
            (inwavfile, outptfile) = [x.strip() for x in line.split(',')];

            # Configure
            sacc_extractor = SAcC.SAcC(config)

            # Apply
            features = sacc_extractor(inwavfile)

            # Write the data out
            sacc_write(outptfile, features, config)
            
            # Maybe wind on utterance counter
            if config['incr_utt'] != 0:
                config.start_utt += 1


# Run the main function if called from the command line
if __name__ == "__main__":
    import sys
    main(sys.argv)

