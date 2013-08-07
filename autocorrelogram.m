function [ac, sc, lags] = autocorrelogram(x, sr, maxlags, h, w)
%    
% x is a input signal consisting of multiple rows, each a separate
% channel, and sr is samplingrate.  
% Using w sec windows at every h-sec length of frame, 
% calculate autocorrelation ac and its energy sc of  the input signal.  
%
% kslee@ee.columbia.edu, 6/16/2005
%
if nargin < 3
    maxlags = 400;
end
if nargin < 4
    h = 0.01; % 10ms, frame length
end
if nargin < 5
    w = 0.025; % 25ms, window length
end

% Change time into the length of frame and window
frmL = sr*h;
winL = sr*w;

%maxlags = min(winL,maxlags);
lags = [1:maxlags];

% the channels and points of input signal
[nchs, npts] = size(x);

% the total number of frames within each segment
% nfrms = floor((npts-(2*winL-frmL))/frmL);
%% [2011-12-14] Byung Suk Lee bsl@ee.columbia.edu
%nfrms = 1 + floor((npts - winL)/frmL);
%% 2012-06-13 dpwe - to avoid segfault in mex
nfrms = 1 + floor((npts + 1 - winL - maxlags)/frmL);
% line 61 of autocorr.c is 
% 		z1 += xp[base+winL+j]*xp[base+winL+j+eta]; 
% where base = (f-1)*frmL;
% and f = 0..(nfrms-1)
% and j = 0..(winL-1)
% and eta = 0..(maxlags-1)
%  and the index must stay within npts


%nac = zeros(nchs, maxlags, nhops);
ac = zeros(nchs, maxlags, nfrms);
sc = zeros(nchs, maxlags, nfrms);

%tic;

for ch = 1:nchs
    
    xx = x(ch,:);
    
%    disp(['nfrms=',num2str(nfrms),' len(xx)=',num2str(length(xx))]);
    [c, s]= autocorr(xx,frmL,nfrms,maxlags,winL);
      
    for nf = 1:nfrms
        ac(ch,:,nf) = c((nf-1)*maxlags+(1:maxlags));
        sc(ch,:,nf) = s((nf-1)*maxlags+(1:maxlags));
    end
    
end

