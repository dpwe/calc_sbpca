function [subbands,freqs] = sbpca_subbands(d,sr,params)
% [subbands,freqs] = sbpca_subbands(d,sr,params)
%   Filter into subbands for sbpca
%   freqs returns the center frequencies for each subband.
%   subbands is <nchs = 24 x ntime == length(d)>
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

postpad = 0.04;  % as in sub_cal_ac

%subbands = filterbank(fbank.b, fbank.a, ...
%                      [dithering(d)', zeros(1,round(postpad*sr))], ...
%                      N, SQ, fbank.t);

%function M = filterbank(B,A,X,N,SQ,T)
X = [dithering(d)', zeros(1,round(postpad*sr))];

% recover number of filters
bands = size(params.fbank.b,1);

% find size of X
xsize = length(X);

% initialize output array to full size
% transpose domain  - avoids quite so much swapping during inner loop
subbands = zeros(bands,xsize);
% normal domain
%M = zeros(bands,floor((xsize+N-1)/N));

% calculate each row
for filt = 1:bands
%  disp(['band ' int2str(filt)]);
  % pad t zeros on the end, since we're going to chop from tail
  t = round(params.fbank.t(filt));
  y = filter(params.fbank.b(filt, :), ...
             params.fbank.a(filt, :), ...
             [X,zeros(1, t)]);
  % shift the output to discard the first <t> samples
  y = y((t+1):end);
  % rectify the signal
  subbands(filt,:) = max(y,0);
end

freqs = params.fbank.cf;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xo] = dithering(x)

% Ensure consistent random sequence
rand('state', 0);

xlen = length(x);
dither = rand(xlen,1) + rand(xlen,1) - 1;
spow = std(x);
xo = x + 1e-6 * spow * dither;
%disp('dithering off')
%xo = x;
