function output = mlp_apply(mlp, input)
% output = mlp_apply(mlp, input)
%    Apply the MLP defined in the <mlp> structure to the columns of
%    data in <input>.  So far, MLP is always simple 3 layer.
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca refactor cleanup

% How many columns of input do we have?
nfrms = size(input,2);

% Normalize with offsets and scales
Dn = diag(mlp.sca)*(input - repmat(mlp.ofs,1,nfrms));

% the first layer
hl = (mlp.IH'*Dn + repmat(mlp.HB,1,nfrms));
hlo = 1 ./ (1 + exp(-hl));

% the second layer
ol = (mlp.HO'*hlo + repmat(mlp.OB,1,nfrms));
ole = exp(ol);
output = ole ./ repmat(sum(ole),size(ole,1),1);

% figure, imagesc(olo'), colormap(1-gray), axis xy, grid on
