function vqs = sbpca_vqs(pcas, params)
% vqs = sbpca_vqs(pcas, params)
%    Convert subband principal components to vector quantized values 
%    for SBPCA.
%    pcas are (pcadim == 10, chans == 24, timefrm)
%    vqs are (blocks == 4, timefrm)
%    each value is one VQ index (0..1023).
% 2013-08-03 Dan Ellis dpwe@ee.columbia.edu


[nRec, nFtr, K] = size(params.vqcodebook); % 4 x 60 x 1000
[nDim, nChs, nTim] = size(pcas); % 10 x 24 x T

% Reshape pcas into vectors to be quantized
grouping = nChs / nRec; % better hope it's 24 / 4 = 6
pcasr = permute(pcas, [2 1 3]);
% now pcasr = nCh x nDim x nTim

tmp = zeros(grouping*nDim, nRec, nTim);

for i = 1:nRec
  CBrec.codewords = squeeze(params.vqcodebook(i,:,:))';
  CBrec.mean = params.vqmeans(i,:);
  CBrec.std = params.vqstds(i,:);
  chs = (i-1)*grouping + [1:grouping];
  rpcasr = reshape(pcasr(chs,:,:), grouping*nDim, nTim);
  vqs(:,i) = acVQQuantize(rpcasr', CBrec);
  tmp(:, i, :) = rpcasr;
end

vqs = vqs';

%vqs = tmp;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I = acVQQuantize(D,CB)
% I = acVQQuantize(D,CB)
%    Return I as the quantization of each data row I according to
%    the codebook CB.mean, CB.std, CB.codewords.
% 2012-07-31 Dan Ellis dpwe@ee.columbia.edu

nD = size(D,1);
maxdrows = min(10000, nD);
nblocks = ceil(nD/maxdrows);

I = zeros(nD,1);

if nD > 0

  cw = CB.codewords';
  ncw = size(cw, 2);
  CBmag = sum(cw .* cw, 1);
  CBmagmat = repmat(CBmag, maxdrows, 1);
  CBmean = repmat(CB.mean, maxdrows, 1);
  CBscal = repmat(1./CB.std, maxdrows, 1);

  for b = 1:nblocks
    bkix = ((b-1)*maxdrows+1):min(nD,b*maxdrows);
    nbk = length(bkix);
    if nbk < maxdrows
      % truncate the repmat'd magmat on the last iteration
      CBmagmat = repmat(CBmag, nbk, 1);
      CBmean = repmat(CB.mean, nbk, 1);
      CBscal = repmat(1./CB.std, nbk, 1);
    end

    % Extract rows and apply mean/var normalization
    Dblock = (D(bkix, :) - CBmean) .* CBscal;

    % find distances to each codeword
  %  dd = my_sqdist(C.codewords, Dblock);
    pmag = sum(Dblock .* Dblock, 2);
    dd = CBmagmat + repmat(pmag, 1, ncw) - 2*Dblock*cw;

    % Sort the columns (distances to each codeword) to find closest..
    [dds,Ib] = min(dd,[],2);

    % Save
    I(bkix') = Ib;

  end

end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function m = my_sqdist(p, q)
% % SQDIST      Squared Euclidean or Mahalanobis distance.
% % SQDIST(p,q)   returns m(i,j) = (p(i,:) - q(j,:))*(p(i,:) - q(j,:))'.
%
% %  From Tom Minka's lightspeed toolbox
%
% [pn,d] = size(p);
% [qn,d] = size(q);
%
% pmag = sum(p .* p, 2);
% qmag = sum(q .* q, 2);
% m = repmat(qmag', pn, 1) + repmat(pmag, 1, qn) - 2*p*q';
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
