function posteriors = sbpca_classif(pcas, params)
% posteriors = sbpca_classif(pcas, params)
%     Apply the MLP classifier from params to the pcas features
%     passed in, return the outputs, which are interpreted as
%     per-pitch-category posteriors.
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca refactor cleanup

%  [obs] = mlp_fwd(D, net.IH', net.HO', net.HB', net.OB', net.ofs, net.sca);

