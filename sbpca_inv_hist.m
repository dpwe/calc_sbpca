function vqs = sbpca_inv_hist(hists, params)
% vqs = sbpca_inv_hist(hists, params)
%   Undo the collapsing of VQ bins into histograms ... ?
% 2013-08-11 Dan Ellis dpwe@ee.columbia.edu rebuild of calcSBPCA

% One way to do this would be to make an n-gram model of VQ
% codewords then tweak the priors of each codeword based on the
% histogram for a particular segment, then draw from it.  You could
% use the n-gram to provide consistency across boundaries too.
