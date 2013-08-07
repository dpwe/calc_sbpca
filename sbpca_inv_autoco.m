function subbands = sbpca_inv_autoco(autocos, params)
% subbands = sbpca_inv_autoco(autocos, params)
%   Invert the short-time autocorrelation in sbpca
%   autocos  is nlag x nchs x nframes
%   subbands is nchs x ntime
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

% This is the hard bit.  We lost the phase...

[nlag, nchs, nframes] = size(autocos);

dur = nframes*params.hoptime;
nframes = size(autocos,3);

hoppts = round(params.hoptime * params.sr);
winpts = round(params.wintime * params.sr);

win2pts = 2*nlag;
ywin = [hann(win2pts-1)',0];

alen = round(params.sr * dur) + win2pts;

subbands = zeros(nchs, alen);

for ch = 1:nchs

  % Convert autocos into STFTs
  ac = squeeze(autocos(:,ch,:));
  ffts = fft([ac;zeros(1, size(ac,2));ac(end:-1:2,:)]);
  
  % incrementally reconstruct each window based on the phase of the overlap
  out = zeros(1,alen);
  
  for j = 1:nframes
    
    xx = (j-1)*hoppts + [1:win2pts];
    
    yi = out(xx);
    YI = fft(yi);
    
    YO = abs(ffts(:,j)') .* exp(sqrt(-1)*angle(YI));
    yo = real(ifft(YO));
    
    % cross fade
    yo = ywin .* yo;
    out(xx) = out(xx) + yo;
  end
  
  subbands(ch, :) = out;
end
