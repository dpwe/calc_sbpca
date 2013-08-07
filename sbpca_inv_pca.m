function autocos = sbpca_inv_pca(pcas, params)
% autocos = sbpca_inv_pca(pcas, params)
%    Undo the effect of taking PCA of subband autocos
%    for SBPCA.
%    pcas is ndims x nchs x nfrms
%    autocos  is nlag x nchs x nframes
% 2013-08-03 Dan Ellis dpwe@ee.columbia.edu

[ndims, nchs, nfrms] = size(pcas);

% params.mapping{ch} is nlags x ndims
% and is an orthonormal output of pca
% so it's self-inverse or something

[nlag, ndim] = size(params.mapping{1});

autocos = zeros(nlag, nchs, nfrms);

for c = 1:nchs
  % invmapping is nlags x ndims
  autocos(:, c, :) = params.mapping{c} * squeeze(pcas(:, c, :));
end
