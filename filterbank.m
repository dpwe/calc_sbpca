function M = filterbank(B,A,X,N,SQ,T)
% M = filterbank(B,A,X,N,SQ,T)             Apply an IIR filterbank to a signal.
%	Takes the filterbank defined by B and A, where each row is 
%	the coefficients to an IIR filter, and applies each to X, 
%	generating a matrix output M whose each row is a filtered 
%	version of X using a different filter.  If N is specified 
%       and greater than zero, the outputs are half-wave rectified
%       (negative values set to zero).  If N is
%	greater than 1, the (rectified) outputs are smoothed and 
%	subsampled by that factor.  If SQ is present and nonzero, and N>0 
%	perform the smoothing on the squared outputs, then root again.
%	T, if present, is a vector of *delays* for each filter, which 
%	are to be compensated-out by discarding the first few samples 
%	of the corresponding channel.
% dpwe 1994jun21.  Uses subsmooth.m (and built-in 'filter')

RECT = 1;
if((nargin<4) | (N==0))  N = 1; RECT=0;  end
if(nargin<5)  SQ = 0; end
if(nargin<6)  T = 0;  end

% recover number of filters
s = size(A);
bands = s(1);  % number of rows in A

if(vsize(T)<bands)
  T = T(1) * ones(1, bands);
end

% find size of X
xsize = max(size(X));

% initialize output array to full size
% transpose domain  - avoids quite so much swapping during inner loop
M = zeros(floor((xsize+N-1)/N),bands);
% normal domain
%M = zeros(bands,floor((xsize+N-1)/N));

% calculate each row
for filt = 1:bands
%  disp(['band ' int2str(filt)]);
  % get filter parameters
  a = A(filt, :);
  b = B(filt, :);
  t = round(T(filt));	% samples to shift must be an integer
  % pad t zeros on the end, since we're going to chop from tail
  y = filter(b,a,[X,zeros(1,t)]);
  ly = vsize(y);
  % shift the output to discard the first <t> samples
  y = y([(t+1):ly]);
  if RECT > 0
    if SQ == 0
      % rectify the signal before smoothing it
      y = max(y,0);
      if(N>1)      
        y = subsmooth(y, N, 2*N+1);
      end	
    else % SQ nonzero so do in squared domain
      y = y.*y;
      if(N>1)
        y = subsmooth(y, N, 2*N+1);
      end	
      y = sqrt(y);
    end;
  end;
%  M(filt,:) = y;
  M(:,filt) = y';
end

% if transpose domain
M=M';
