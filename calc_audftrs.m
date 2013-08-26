function [hists, params] = calc_audftrs(d, sr)
% [hists, params] = calc_audftrs(d, sr)
%      Calculate "auditory model" subband PCA VQ histogram feature.
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu

% Main subband PCA core
[pcas,params] = calc_sbpca(d, sr);

% Load VQ codebook
vqfile = 'CB-sbpca-4x60x1000.mat';
C = load(vqfile);
params.vqcodebook = C.codebook;
params.vqmeans = C.recMean;
params.vqstds = C.recStd;

% vector quantize - <nRect> x <nframe>
VQs = sbpca_vqs(pcas, params);

% histogram - <nRect x cbSize> x <nblock>
hists = sbpca_hist(VQs, params);
% 2013-08-11 Up to here verified as almost identical to calcSBPCA
% (just off by one or two frames on histogram aggregation windows)
% but I think these ones are more on-target (ignoring the 25 ms
% flank, I suppose).
