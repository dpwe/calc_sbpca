function [forward, feedback, Tds, BW]=MakeERBFilter(fs,cf,CQ)
% [B,A,Tds,BW] = MakeERBFilter(fs,cf,CQ)     Design a Patterson cochlea filter.
% 	Computes the filter coefficients for a single Gammatone filter.
%	The filters were defined by Patterson and Holdworth for simulating 
% 	the cochlea.  The results are returned as arrays of filter
% 	coefficients.  Each row of the filter arrays (forward and feedback)
% 	can be passed to the MatLab "filter" function, or you can do all
% 	the filtering at once with the ERBFilterBank() function.
%	Tds is a vector of group delays in samples for each filter.
%	BW is the bandwidth (in Hz) for each filter.
% 	The filter bank contains filters for each frequency in <cf>
%	If CQ is present and nonzero, force the filters to remain 
%	const Q right down into the LF.
% dpwe, after malcolm 1994sep19

if(nargin < 3)	CQ = 0;	end;

T=1/fs;
% Change the following parameters if you wish to use a different
% ERB scale.
EarQ = 9.26449;               %  Glasberg and Moore Parameters
minBW = 24.7;
order = 1;

% All of the following expressions are derived in Apple TR #35, "An
% Efficient Implementation of the Patterson-Holdsworth Cochlear
% Filter Bank."
%% cf = -(EarQ*minBW) + exp((1:numChannels)'*(-log(fs/2 + EarQ*minBW) + ...
%%                            log(lowFreq + EarQ*minBW))/numChannels) ...
%%                          *(fs/2 + EarQ*minBW);
if(CQ==0)
  ERB = ((cf/EarQ).^order + minBW^order).^(1/order);
else
  % True-CQ hack - for better phase alignment of filters
  ERB = (cf/EarQ)*((fs/8)/EarQ+minBW)*EarQ/(fs/8);
end
B=1.019*2*pi*ERB;

% Below here, just cf, T and B used

gain = abs((-2*exp(4*i*cf*pi*T)*T + ...
		 2*exp(-(B*T) + 2*i*cf*pi*T).*T.* ...
		 (cos(2*cf*pi*T) - sqrt(3 - 2^(3/2))* ...
		  sin(2*cf*pi*T))) .* ...
           (-2*exp(4*i*cf*pi*T)*T + ...
             2*exp(-(B*T) + 2*i*cf*pi*T).*T.* ...
              (cos(2*cf*pi*T) + sqrt(3 - 2^(3/2)) * ...
               sin(2*cf*pi*T))).* ...
           (-2*exp(4*i*cf*pi*T)*T + ...
             2*exp(-(B*T) + 2*i*cf*pi*T).*T.* ...
              (cos(2*cf*pi*T) - ...
               sqrt(3 + 2^(3/2))*sin(2*cf*pi*T))) .* ...
           (-2*exp(4*i*cf*pi*T)*T + 2*exp(-(B*T) + 2*i*cf*pi*T).*T.* ...
           (cos(2*cf*pi*T) + sqrt(3 + 2^(3/2))*sin(2*cf*pi*T))) ./ ...
          (-2 ./ exp(2*B*T) - 2*exp(4*i*cf*pi*T) +  ...
           2*(1 + exp(4*i*cf*pi*T))./exp(B*T)).^4);
feedback=zeros(length(cf),9);
forward=zeros(length(cf),5);
forward(:,1) = T^4 ./ gain;
forward(:,2) = -4*T^4*cos(2*cf*pi*T)./exp(B*T)./gain;
forward(:,3) = 6*T^4*cos(4*cf*pi*T)./exp(2*B*T)./gain;
forward(:,4) = -4*T^4*cos(6*cf*pi*T)./exp(3*B*T)./gain;
forward(:,5) = T^4*cos(8*cf*pi*T)./exp(4*B*T)./gain;
feedback(:,1) = ones(length(cf),1);
feedback(:,2) = -8*cos(2*cf*pi*T)./exp(B*T);
feedback(:,3) = 4*(4 + 3*cos(4*cf*pi*T))./exp(2*B*T);
feedback(:,4) = -8*(6*cos(2*cf*pi*T) + cos(6*cf*pi*T))./exp(3*B*T);
feedback(:,5) = 2*(18 + 16*cos(4*cf*pi*T) + cos(8*cf*pi*T))./exp(4*B*T);
feedback(:,6) = -8*(6*cos(2*cf*pi*T) + cos(6*cf*pi*T))./exp(5*B*T);
feedback(:,7) = 4*(4 + 3*cos(4*cf*pi*T))./exp(6*B*T);
feedback(:,8) = -8*cos(2*cf*pi*T)./exp(7*B*T);
feedback(:,9) = exp(-8*B*T);

% from differentiating the envelope function, t^(n-1)exp(-t/wb)
n = 4;
Tds = fs*(n-1)./B;
BW = ERB;
