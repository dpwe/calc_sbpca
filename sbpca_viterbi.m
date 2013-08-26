function [pitches,transmat,posteriors] = sbpca_viterbi(posteriors, params)
% pitches = sbpca_viterbi(posteriors, params)
%    Find the best (viterbi) path through a set of pitch class
%    posteriors, for the SAcC pitch tracker.
%    <posteriors> is <nbins> x <nframes>
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca refactor cleanup

% Load the bin -> Hz mapping
pitchfile = 'pitch_candidates_freqz.txt';
pitchmap = [0; textread(pitchfile, '%f')];

% Equalizing variance in log-posterior domain per BSL implementation
posteriors = exp(standardize(log(posteriors)));

% Set up data for decode
[nbins, nframes] = size(posteriors);
npch = nbins - 1; % number of actual pitches (i.e., all except unvoiced)

% Parameters
uvtrp = 0.9;  % prob of going from unvoiced to voiced (9x larger
              % than BSL's code, to compensate for normalization of txmat)
vutrp = 0.01; % prob of going from voiced to unvoiced
transfloor = exp(-10.0); % smallest transition probability
wdyn = 3.0;   % laplacian half-width for transition probs
vp = 0.9;     % scaling of unvoiced state

% Transition matrix - row = from, column = to
% A matrix of how far apart two bins are
ijdiff = abs(repmat([1:npch]',1,npch) - repmat([1:npch],npch,1));
% pitch-to-pitch transitions are laplacian
%pptxmat = max(transfloor, exp(-abs(ijdiff)/wdyn));
pptxmat = log(transfloor + exp(exp(-abs(ijdiff)/wdyn)));
% normalize rows of pitch-to-pitch transitions to be true probabilities
pptxmat = pptxmat ./ repmat(sum(pptxmat,2),1,npch);
% transmat wraps unvoiced state around pitch-to-pitch
transmat = [(1-uvtrp), uvtrp*ones(1, npch)/npch; ...
            vutrp*ones(npch, 1), (1-vutrp)*pptxmat];

% penalize unvoiced posterior & renormalize
posteriors(1,:) = vp * posteriors(1,:);
posteriors = posteriors ./ repmat(sum(posteriors), nbins, 1);
priors = ones(nbins,1)/nbins;

path = viterbi_path(posteriors, priors, transmat);

% Map to pitches
pitches = pitchmap(path);


%%%%%%%%%%%%%%%%%%%%%%%
function N = standardize(A)
% Make each column of an array have a zero mean and unit sd
% was "normalise" by kslee@ee.columbia.edu (not to confuse with kpm's normalise)
ndim = size(A,1);
s = std(A);
% normalize each column
N = (A - repmat(mean(A),ndim,1))./repmat(s+(s==0),ndim,1);
