function [subbands,freqs] = sbpca_subbands(d,sr,params)
% [subbands,freqs] = sbpca_subbands(d,sr,params)
%   Filter into subbands for sbpca
%   freqs returns the center frequencies for each subband.
%   subbands is <nchs = 24 x ntime == length(d)>
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

persistent fbank
if isempty(fbank)
  bpo = 6; nchs = 24;
  fmin = 100; fb_q = 8; ord = 2; ftype = 2;
  [fbank.b, fbank.a, fbank.t, w, fbank.cf] ...
      = bpfiltbank(sr, fmin, bpo, nchs, fb_q, ord, ftype);
%  subplot(111)
%  for i = 1:size(fbank.b,1)
%    [hh,ww] = freqz(fbank.b(i,:),fbank.a(i,:));
%    plot(ww/pi, 20*log10(abs(hh)));
%    hold on;
%  end
%  hold off;
end

N = 1;
SQ = 0;
postpad = 0.04;  % as in sub_cal_ac

subbands = filterbank(fbank.b, fbank.a, ...
                      [dithering(d)', zeros(1,round(postpad*sr))], ...
                      N, SQ, fbank.t);
freqs = fbank.cf;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xo] = dithering(x)

% Ensure consistent random sequence
rand('state', 0);

xlen = length(x);
dither = rand(xlen,1) + rand(xlen,1) - 1;
spow = std(x);
xo = x + 1e-6 * spow * dither;

