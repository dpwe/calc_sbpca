function [pitches, pvx, params] = calc_sacc(d, sr)
% [pitches, pvx, params] = calc_sacc(d, sr)
%      Calculate sbpca-based pitch tracker, sacc
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu

% Main subband PCA core
[pcas, params] = calc_sbpca(d, sr);

[ndim, nsubband, nframe] = size(pcas);

% Load classifier network
mlpfile = 'sub_qtr_rats_keele_sr8k_bpo6_sb24_k10_ep5_h100.wgt';
params.mlpwts = mlp_load(mlpfile);

% mlp_apply expects each input pattern as a column
% MLP expects 24 subbands x 10 pcas raveled with the subband as the
% fastest-changing dimension i.e. 24 PCA0s, then 24 PCA1s, then 24 PCA2s...
posteriors = mlp_apply(params.mlpwts, ...
                       reshape(permute(pcas, [2 1 3]), ...
                               nsubband*ndim, nframe));

% Top row of posteriors is prob(nonspeech), so complement it to get Pr(Vx)
pvx = 1 - posteriors(1,:);   % == sum(posteriors(2:end,:))

% Run the viterbi path to find a pitch track
pitches = sbpca_viterbi(posteriors, params); % keele PTE=10.4%, same as original SAcC

