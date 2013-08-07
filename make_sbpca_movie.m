function make_sbpca_movie(d,sr,moviename)
% make_autoco_movie(d,sr,moviename)
%    Make a movie of the subband autocorrelation for the specified
%    sound. 
% 2013-05-27 Dan Ellis dpwe@ee.columbia.edu

%frameRate = 25;
%mvo = avifile(moviename,'fps',frameRate);
%mvo = addframe(mvo,framecdata);

MakeQTMovie('start',moviename);

params.wintime = 0.025;
params.hoptime = 0.010;
params.sr = 8000;
params.maxlag = round(params.wintime * params.sr);

subbands = sbpca_subbands(d,sr,params);
autocos = sbpca_autoco(subbands, params);

framerate = 25;
frametime = 1/framerate;

dur = length(d)/sr;

nframes = floor(dur/frametime);

rgcolormap();

for frame = 1:nframes
  ftime = (frame-1)*frametime;
  aframe = 1+round(ftime/params.hoptime);
  if aframe < size(autocos,3)
    imgsc(squeeze(autocos(:,:,aframe))); title(num2str(ftime));
    MakeQTMovie('addfigure');
  end
end

MakeQTMovie('framerate', framerate);

MakeQTMovie('addsound', d, sr);

MakeQTMovie('finish');
