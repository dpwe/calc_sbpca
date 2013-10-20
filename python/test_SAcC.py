#!/usr/bin/env python
"""
test_SAcC.py
test harness for SAcC.py
"""

import SAcC
import numpy as np
config = {}

# Input/Output files
filename    = 'input8k.wav'
#filename    = '60sec_37497_20120228_020100_24537_appen-cts-pus_F.wav'
outfilename = 'sacc8k.txt'
#outfilename = '60sec_37497_20120228_020100_24537_appen-cts-pus_F.txt'
#outsphname  = 'sacc8k.sph'
#outmatname  = 'sacc8k.mat'

# Add config parameters to dictionary
# diff file for py
config['pca_file']    = 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat' 
#config['wgt_file']    = 'aux/rats_sr8k_bpo6_sb24_k10_aCH_h100.wgt'
#config['norms_file']  = 'aux/tr_rats_sr8k_bpo6_sb24_k10.norms'
config['wgt_file']    = 'aux/sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt'
config['norms_file']  = 'aux/tr_keele_rbf_pinknoise_sr8000_bpo6_nchs24_k10.norms'
config['pcf_file']    = 'aux/pitch_candidates_freqz.txt'
#config['kdim']        = 10   # not used - inferred from pca file
config['nchs']        = 24
config['n_s']         = 1.0  # secs per process block, controls blockframes
config['SBF_sr']      = 8000
config['SBF_fmin']    = 100
config['SBF_bpo']     = 6
config['SBF_q']       = 8  # not actually used for SlanPat ERB filters
config['SBF_order']   = 2  # not actually used for SlanPat ERB filters
#config['SBF_ftype']   = 2  # not used in Python - only supported type
#config['force_mono']  = 1  # stereo downmixing not supported
config['twin']      = 0.025  # New option in Python - autoco window len
config['thop']      = 0.010  # New option in Python - autoco hop
#config['ac_maxlags']  = 200 # not used - inferred from mapping file
#config['hid']         = 100 # not used - inferred from wgt file
#config['npcf']        = 67  # not used - inferred from wgt file
#config['nmlp']        = 68  # not used - inferred from wgt file
config['hmm_vp']      = 0.9 # interpretation changed c/w Matlab
config['write_rownum'] = 0 # prepend row number
config['write_time']  = 1  # prepend time in seconds to output
config['write_sbac'] = 0   # output raw autocorrelations (big - 24 x 200)
config['write_sbpca'] = 0  # output subband pcas (24 x 10)
config['write_posteriors'] = 0 # output raw pitch posteriors (68)
config['write_pitch'] = 1  # output the actual pitch value in Hz (1)
config['write_pvx'] = 1    # output just 1-posterior(unvoiced) (1)
# Tricks with segmenting utterances not implemented in Python
#config['start_utt'] = 0    # what utterance number to start at
#config['incr_utt'] = 0     # increment the utterance each seg (?)
#config['segs_per_utt'] = 1 # break each utterance into this many segs
config['verbose'] = 0


import array as ARRAY
import ctypes

def write_features_in_sph(filename, data, feature_names="SAcC"):
    num_frames, num_elements = np.shape(data)
    shead = "NIST_1A\n   1024\nfile_type -s11 featurefile\n"
    shead += "feature_names -s" + str(len(feature_names)) 
    shead += " " + feature_names + "\n"
    shead += "num_elements -i " + str(int(num_elements)) + "\n"
    shead += "num_frames -i " + str(int(num_frames)) + "\n"
    shead += "sample_count -i " + str(int(num_elements * num_frames)) + "\n"
    shead += "end_head\n"
    shead += (1024 - len(shead)) * " "
    f = open(filename, 'wb')
    buf = ctypes.create_string_buffer(shead, 1024)
    f.write(buf)
    v = ARRAY.array('f')
    v.fromlist(list(data.ravel()))
    v.byteswap()
    v.tofile(f)
    f.close()

# Construct the extractor then extract the features for [filename]
extractor = SAcC.SAcC(config)
features  = extractor(filename)

# Save features in ascii format
np.savetxt(outfilename, features, fmt='%f', delimiter=' ', newline='\n')
#write_features_in_sph(outsphname, features)
#scipy.io.savemat(outmatname, {'features': features})
