function pcas = sbpca_pca(autocos, params)
% pcas = sbpca_pca(autocos, params)
%    Convert subband autocorrelations to principal components
%    for SBPCA.
%    autocos  is nlag x nchs x nframes
%    pcas is ndims x nchs x nfrms
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

[nlags, nchs, nfrms] = size(autocos);

pchs = size(params.mapping,1);
[plags, pdims] = size(params.mapping{1});

pcas = zeros(pdims,nchs,nfrms);
for c = 1:nchs
  tmpac = squeeze(autocos(:,c,:));
  %% To ignore frames that contain NaN elements
  sumtmpac = sum(tmpac);
  tmpac(:,isnan(sumtmpac)) = 0;
  pcas(:,c,:) = params.mapping{c}' * tmpac;
end


