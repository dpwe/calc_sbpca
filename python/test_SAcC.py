#!/usr/bin/env python

from SAcC import *
import numpy as np
config = {}

# Input/Output files
filename    = 'input.wav'
outfilename = 'sacc.txt'
outsphname  = 'sacc.sph'

# Add config parameters to dictionary
config['pca_file']    = 'aux/mapping-pca_sr8k_bpo6_sb24_k10.mat' # diff file for py
#config['wgt_file']    = 'aux/rats_sr8k_bpo6_sb24_k10_aCH_h100.wgt'
#config['norms_file']  = 'aux/tr_rats_sr8k_bpo6_sb24_k10.norms'
config['wgt_file']    = 'aux/sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt'
config['norms_file']  = 'aux/tr_keele_rbf_pinknoise_sr8000_bpo6_nchs24_k10.norms'
config['pcf_file']    = 'aux/pitch_candidates_freqz.txt'
#config['kdim']        = 10   # not used - inferred from pca file
config['nchs']        = 24
#config['n_s']         = 10   # not currently used
config['SBF_sr']      = 8000
config['SBF_fmin']    = 100
config['SBF_bpo']     = 6
config['SBF_q']       = 8  # not actually used for SlanPat ERB filters
config['SBF_order']   = 2  # not actually used for SlanPat ERB filters
#config['SBF_ftype']   = 2  # not used in Python - only supported type
#config['force_mono']  = 1  # stereo downmixing not supported
config['ac_win']      = 0.025  # New option in Python - autoco window len
config['ac_hop']      = 0.010  # New option in Python - autoco hop
#config['ac_maxlags']  = 200 # not used - inferred from mapping file
#config['hid']         = 100 # not used - inferred from wgt file
#config['npcf']        = 67  # not used - inferred from wgt file
#config['nmlp']        = 68  # not used - inferred from wgt file
config['hmm_vp']      = 0.9 # interpretation changed c/w Matlab

import array as ARRAY
import ctypes

def write_features_in_sph(filename, data, feature_names="SAcC"):
    num_frames, num_elements = np.shape(data)
    shead = "NIST_1A\n   1024\nfile_type -s11 featurefile\n"
    shead += "feature_names -s" + str(len(feature_names)) + " " + feature_names + "\n"
    shead += "num_elements -i " + str(int(num_elements)) + "\n"
    shead += "num_frames -i " + str(int(num_frames)) + "\n"
    shead += "sample_count -i " + str(int(num_elements * num_frames)) + "\n"
    shead += "end_head\n"
    shead += (1024 - len(shead)) * " "
    f = open(filename, 'wb')
    buf = ctypes.create_string_buffer(shead, 1024)
    f.write(buf)
    v=ARRAY.array('f')
    v.fromlist(list(data.ravel()))
    v.byteswap()
    v.tofile(f)
    f.close()



# Construct the extractor then extract the features for [filename]
extractor = SAcC(config)
features  = extractor(filename)

# Save features in ascii format
np.savetxt(outfilename, features, fmt='%.6f', delimiter=' ', newline='\n')
write_features_in_sph(outsphname, features)
