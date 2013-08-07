function Y = inv_sbpca(d,sr)
% D = inv_sbpca(Y,sr)
%   Invert SBPCA calculation
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

% parameters
params.wintime = 0.025
params.sr = sr;
params.maxlags = round(params.wintime * params.sr);

mapfile = 'pca_sr8k_bpo6_sb24_k10.mat';
M = load(mapfile);
params.mapping = M.mapping;


% histogram
%VQs = sbpca_inv_hist(hists, params);

% vector quantize
%pcas = sbpca_inv_vqs(VQs, params);

% principal components
autocos = sbpca_inv_pca(pcas, params);

% autocorrelate
subbands = sbpca_inv_autoco(autocos, params);

% subbands
[d, sr] = sbpca_inv_subbands(subbands, params);




