function [autocos,acf_engy] = sbpca_autoco(subbands, params)
% [autocos,energy] = sbpca_autocos(subbands,sr)
%   Calculate autocorrelations over subbands for sbpca.
%   subbands is nchs x ntime
%   autocos  is nlag x nchs x nframes
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

sr = params.sr;

hop = params.hoptime;
win = params.wintime;

maxlags = round(win*sr);

% multichannel autocorrelation
[acf, acf_engy, lags] = autocorrelogram(subbands, sr, ...
                                        maxlags, hop, win); 

autocos = acf ./ (acf_engy + (acf_engy == 0));
%disp('no autoco norm')

% Make it lags x subbands x timeframes
autocos = permute(autocos, [2 1 3]);
