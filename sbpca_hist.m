function hists = sbpca_hist(vqs, params)
% hists = sbpca_hist(vqs, params)
%    Collapse the vq codewords into per-segment histograms.
%    Each column of hists is the mean of all the frames in
%    params.histwin seconds every params.histhop seconds.
%
% 2013-08-11 Dan Ellis dpwe@ee.columbia.edu rebuild of sbcpa calculation

framesperblock = round(params.histwin / params.hoptime);
framesperhop   = round(params.histhop / params.hoptime);

base = 0;

vqslen = length(vqs);

nblocks = round(vqslen/framesperhop);

[ncb, ndim, ncw] = size(params.vqcodebook); % 4, (60), 1000

nbins = ncb * ncw;

hists = zeros(nbins, nblocks);

for b = 1:nblocks
  frames = (b-1)*framesperhop;
  frameix = (frames+1):min(frames+framesperblock, vqslen);
  for cb = 1:ncb
    for f = frameix
      ix = (cb-1)*ncw + vqs(cb, f);
      hists(ix, b) = hists(ix ,b) + 1;
    end
  end
  hists(:,b) = hists(:,b) / length(frameix) / ncb;
end

    
  