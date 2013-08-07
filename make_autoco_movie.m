function make_autoco_movie(d,sr,moviename,name,slowdown)
% make_autoco_movie(d,sr,moviename,name,slowdown)
%    Make a movie of the subband autocorrelation for the specified
%    sound. Slow down by factor.
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

if nargin < 4; name = '<unk>'; end
if nargin < 5; slowdown = 1.0; end

if exist('pvoc') ~= 2
  % for slowing down the sound
  addpath('~/projects/pvoc');
end

targetsr = 8000;

if sr ~= targetsr
  d = resample(d,targetsr,sr);
  sr = targetsr;
end

%frameRate = 25;
%mvo = avifile(moviename,'fps',frameRate);
%mvo = addframe(mvo,framecdata);

MakeQTMovie('start',moviename);

params.wintime = 0.025;
params.hoptime = 0.010;
params.sr = sr;
params.maxlags = round(params.wintime * params.sr);

[subbands,freqs] = sbpca_subbands(d,sr,params);
autocos = sbpca_autoco(subbands, params);

framerate = 25;
frametime = 1/framerate;

dur = length(d)/sr;

nframes = floor(dur*slowdown/frametime);

subplot(211)
specgram(d,256,sr,224);
caxis([-60 0]+max(caxis()));
tw = 2.0;
ax = axis();
fmin = ax(3);
fmax = ax(4);

hold on; 
redline = plot([0 0], [fmin fmax], '-r'); 
hold off;

%rgcolor();
gcolor();

tt = [0:size(autocos,2)-1]/sr;
ff = 1:size(autocos,1);

for frame = 1:nframes
  ftime = (frame-1)*frametime/slowdown;
  aframe = 1+round(ftime/params.hoptime);
  if aframe < size(autocos,3)

    subplot(212);
    imgsc(tt,ff,squeeze(autocos(:,:,aframe))); 
    xx = 1:3:length(ff);
    set(gca, 'YTick', xx);
    set(gca, 'YTickLabel', round(freqs(xx)));
    xlabel('lag / s');
    ylabel('freq / Hz');
    
    title(sprintf('%s - %.2f', name, ftime), 'interpreter', 'none');

    subplot(211)
    axis([ftime - tw, ftime + tw, fmin, fmax]);
    hold on; 
    delete(redline);
    redline = plot([ftime ftime], [fmin fmax], '-r'); 
    hold off;
    
    MakeQTMovie('addfigure');
  end
end

MakeQTMovie('framerate', framerate);

if slowdown ~= 1
  nfft = 256;
  d = pvoc(d,1/slowdown,nfft);
end


MakeQTMovie('addsound', d, sr);

MakeQTMovie('finish');

lastdot = [max(find(moviename=='.')), length(moviename)+1];
mp4name = [moviename(1:lastdot(1)-1), '.mp4'];

disp(['ffmpeg -i ',moviename, ...
      ' -vcodec libx264 -preset fast -crf 28 -threads ' ...
      '0 -acodec libfaac -ac 1 -ar 16000 -ab 64k ', mp4name]);
