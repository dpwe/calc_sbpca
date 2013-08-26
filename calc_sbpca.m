function [pcas, params] = calc_sbpca(d, sr)
% [pcas, params] = calc_sbpca(d, sr)
%   Rebuild of SBPCA calculation - up to subband PCA part
%   Output pca has dimensions <pcadims> x <nsubbands> x <nframes>
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

% parameters
params.wintime = 0.025;
params.hoptime = 0.010;
params.sr = sr;

params.histwin = 2.0;
params.histhop = 2.0;

% load the data files
% PCA basis functions
mapfile = 'pca_sr8k_bpo6_sb24_k10.mat';
M = load(mapfile);
params.mapping = M.mapping;

% Setup the filterbank
%bpo = 6; nchs = 24;
%fmin = 100; fb_q = 8; ord = 2; ftype = 2;
%[params.fbank.b, params.fbank.a, params.fbank.t, w, params.fbank.cf] ...
%      = bpfiltbank(params.sr, fmin, bpo, nchs, fb_q, ord, ftype);

params.fbank = sbpca_filterbank(params.sr);

%%%%%%%%%%% Calculation %%%%%%%%%%%%

% subbands - <nch> x <ntime>
subbands = sbpca_subbands(d, sr, params);

% autocorrelate - <nlag> x <nch> x <nframe>
autocos = sbpca_autoco(subbands, params);

% principal components - <nPC> x <nch> x <nframe>
pcas = sbpca_pca(autocos, params);
% 2013-08-07 Up to here now verified as identical to calcSBPCA

