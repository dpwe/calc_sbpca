function fbank = sbpca_filterbank(SR, FMIN, BPO, BANDS, Q, ORD)
% fbank = sbpca_filterbank(SR, FMIN, BPO, BANDS, Q, ORD)
%    Setup the ERB filterbank for sbpca.
% 2013-08-11 Dan Ellis dpwe@ee.columbia.edu adapted from bpfiltbank.m

if nargin < 2; FMIN = 100; end
if nargin < 3; BPO = 6; end
if nargin < 4; BANDS = 24; end
if nargin < 5; Q = 8; end
if nargin < 6; ORD = 2; end

%disp(['bpfiltbak(sr=',num2str(SR),' fmin=',num2str(FMIN), ...
%      ' bpo=',num2str(BPO),' bands=',num2str(BANDS), ...
%      ' Q=',num2str(Q), ' ORD=', num2str(ORD), ')']);

FMAX = FMIN*exp(log(2)*BANDS/BPO);

if (FMIN <= 0) %( || FMAX <= FMIN)
  error('bpfiltbank: must be 0 < FMIN < FMAX (log scaling)');
end

logfreqfactor = log(FMAX/FMIN)/BANDS;
logminfreq    = log(FMIN);
fmax = exp(logminfreq + BANDS * logfreqfactor);

%disp(['Using Slaney-Patterson filterbank, frq=' num2str(FMIN) '..' num2str(fmax) ', bpo=' num2str(BPO)]);

fblen = 2*ORD + 1;
falen = 4*ORD + 1;

fbank.a = zeros(BANDS, falen);
fbank.b = zeros(BANDS, fblen);
fbank.t = zeros(BANDS, 1);
fbank.bw = zeros(BANDS, 1);
fbank.cf = zeros(1,BANDS);

for filter = 1:BANDS
  cf = exp(logminfreq + (filter - 1)*logfreqfactor);
  [b,a,t,bw] = MakeERBFilter(SR, cf);
  fbank.a(filter, :) = a;
  fbank.b(filter, :) = b;
  fbank.t(filter)    = t;
  fbank.bw(filter)   = bw;
  fbank.cf(filter)   = cf;
end

% subplot(111)
% for i = 1:size(fbank.b,1)
%   [hh,ww] = freqz(fbank.b(i,:),fbank.a(i,:));
%   plot(ww/pi, 20*log10(abs(hh)));
%   hold on;
% end
% hold off;
