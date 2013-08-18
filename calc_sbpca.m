function [Y,params] = calc_sbpca(d,sr)
% [Y,params] = calc_sbpca(d,sr)
%   Rebuild of SBPCA calculation
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

% VQ codewords
vqfile = 'CB-sbpca-4x60x1000.mat';
C = load(vqfile);
params.vqcodebook = C.codebook;
params.vqmeans = C.recMean;
params.vqstds = C.recStd;

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

% vector quantize - <nRect> x <nframe>
VQs = sbpca_vqs(pcas, params);

% histogram - <nRect x cbSize> x <nblock>
hists = sbpca_hist(VQs, params);
% 2013-08-11 Up to here verified as almost identical to calcSBPCA
% (just off by one or two frames on histogram aggregation windows)
% but I think these ones are more on-target (ignoring the 25 ms
% flank, I suppose).

Y = hists;
