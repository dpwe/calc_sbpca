function path = viterbi_path(posteriors, priors, transmat)
% path = viterbi_path(posteriors, priors, transmat)
%     Find best path through spectrogram-like posteriors (one
%     column per time frame).  Transmat is row from, column to.
%     Linear probabilities (not log). 
%     Return sequence of state indices.
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca refactor cleanup

[nbins, nframes] = size(posteriors);

% Array to hold traceback
prev = zeros(nbins, nframes);

% <pstate> holds normalized  probability-to-date of landing in this
% state along best path 
pstate = priors.*posteriors(:,1); 
% normalize probs of best path to each state, to avoid underflow
pstate = pstate/sum(pstate);

% now calculate forward
for i = 2:nframes
  % Find most likely combination of previous prob-to-path, and transition
  [pstate, prev(:,i)] = max(transmat' .* (posteriors(:,i)*pstate'), ...
                            [], 2);
  % Renormalize to keep probabilities in a sensible range
  pstate = pstate/sum(pstate);
end

% traceback best precedent matrix to get best path
path = zeros(1,nframes);
% best final state
[vv,path(nframes)] = max(pstate);
% .. and all its predecessors
for i = nframes-1:-1:1
  path(i) = prev(path(i+1), i+1);
end
