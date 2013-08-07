function pcas = sbpca_inv_vqs(vqs, params)
% pcas = sbpca_inv_vqs(vqs, params)
%    Undo the effect of VQing subband PCA blocks
%    for SBPCA.
%    vqs is nblk x nfrms
%    pcas is ndims x nchs x nfrms
% 2013-08-03 Dan Ellis dpwe@ee.columbia.edu

[nblk, nfrms] = size(vqs);  % 4 x T
[nRec, nFtr, K] = size(params.vqcodebook); % 4 x 60 x 1000

nChs = length(params.mapping);
grouping = nChs / nRec;
nDim = nFtr / grouping;

pcas = zeros(nDim, nChs, nfrms);

for r = 1:nRec
  vqcb = shiftdim(params.vqcodebook(r,:,:),1) ...
         .* repmat(params.vqstds(r, :)', 1, K) ...
         + repmat(params.vqmeans(r, :)', 1, K);
  cws = permute(reshape(vqcb(:,vqs(r,:)), grouping, nDim, nfrms), ...
                [2 1 3]);
  pcas(:, (r-1)*grouping + [1:grouping], :) = cws;
end
