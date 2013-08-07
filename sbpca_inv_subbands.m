function [d, sr] = sbpca_inv_subbands(subbands, params)
% [d, sr] = sbpca_inv_subbands(subbands, params)
%    Invert mapping to subbands for SBPCA.
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

persistent fbank
if isempty(fbank)
  bpo = 6; nchs = 24;
  fmin = 100; fb_q = 8; ord = 2; ftype = 2;
  [fbank.b, fbank.a, fbank.t, w, cf] ...
      = bpfiltbank(params.sr, fmin, bpo, nchs, fb_q, ord, ftype);
%  subplot(111)
%  for i = 1:size(fbank.b,1)
%    [hh,ww] = freqz(fbank.b(i,:),fbank.a(i,:));
%    plot(ww/pi, 20*log10(abs(hh)));
%    hold on;
%  end
%  hold off;
end

d = ifilterbank(fbank.b, fbank.a, subbands, 1, fbank.t);

d = d';
sr = params.sr;

