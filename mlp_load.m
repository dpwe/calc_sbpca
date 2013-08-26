function mlp = mlp_load(wtsfile, normsfile)
% mlp = mlp_load(wtsfile, normsfile)
%    Read a multi-layer perceptrion (MLP) definition file from
%    <wtsfile> (and corresponding feature normalization parameters
%    from <normsfile>, default <wtsfilestem>.norms) and return them
%    in an MLP structure, suitable for mlp_apply.
% 2013-08-23 Dan Ellis dpwe@ee.columbia.edu sbpca simplified rewrite.

%[net.IH, net.HO, net.HB, net.OB] = readmlpwts(P.wgt_file,insize,P.hid,P.nmlp);
%[net.ofs, net.sca] = readmlpnorms(P.norms_file,insize);

if nargin < 2; normsfile = ''; end

if length(normsfile) == 0
  [p,n,e] = fileparts(wtsfile);
  normsfile = fullfile(p, [n,'.norms']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Read in the norms file

% We do this first, to figure out the input layer size

fid = fopen(normsfile, 'r');
if (fid == -1)  
  error(sprintf('readnorms: unable to read %s\n', normsfile));
end

% Now read ofs
s = fscanf(fid, '%4s', 1);
if ~strcmp(s,'vec')
  error(sprintf('readnorms: header of "%s" is not "vec" - invalid format\n', s));
end
ihsize = fscanf(fid, '%d\n', 1);
%if ihsize ~= I
%  error(sprintf('readnorms: ofs input size of %d is not I(%d)\n', ihsize, I));
%end
% Accept vector length as input layer size
I = ihsize;
mlp.ofs = fscanf(fid, '%f\n', I);

% Now read sca
s = fscanf(fid, '%4s', 1);
if ~strcmp(s,'vec') 
  error(sprintf('readnorms: 2nd header of "%s" is not "vec" - invalid format\n', s));
end
ihsize = fscanf(fid, '%d\n', 1);
if ihsize ~= I
  error(sprintf('readnorms: sca input size of %d is not I(%d)\n', ihsize, I));
end
mlp.sca = fscanf(fid, '%f\n', I);

fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the weights file

fid = fopen(wtsfile, 'r');
if (fid == -1)  
  error(['unable to read ', wtsfile]);
end

% read the file
s = fscanf(fid, '%8s', 1);
if(s~='weigvec') 
  error(['header of "',s,'" is not "weigvec" - invalid format']);
end
ihsize = fscanf(fid, '%d', 1);
%if ihsize ~= I*H
%  error(['input-hidden size of ',num2str(ihsize),' is not I(',num2str(I),')xH(',num2str(H),')']);
%end
% Infer H from ihsize and I
H = ihsize/I;
mlp.IH = fscanf(fid, '%f', [I,H]);

% Now read 2nd weigvec
s = fscanf(fid, '%8s', 1);
%fprintf(1, 's2 = "%s"\n', s);
if(s~='weigvec') 
  error(['2nd header of "',s,'" is not "weigvec" - invalid format']);
end
hosize = fscanf(fid, '%d', 1);
%if hosize ~= H*O
%  error(['hidden-output size of ',num2str(hosize),' is not H(',num2str(H),')xO(',num2str(O),')']);
%end
% Infer O size from hosize
O = hosize/H;
mlp.HO = fscanf(fid, '%f', [H,O]);

% Now read biasvecs
s = fscanf(fid, '%8s', 1);
if(s~='biasvec') 
  error(['1st bias header of "',s,'" is not "biasvec" - invalid format']);
end
hbsize = fscanf(fid, '%d', 1);
if hbsize ~= H
  error(['hidden bias size of ',num2str(hbsize),' is not H(',num2str(H),')']);
end
mlp.HB = fscanf(fid, '%f', hbsize);

% Finally, second biasvec
s = fscanf(fid, '%8s', 1);
if(s~='biasvec') 
  error(['2nd bias header of "',s,'" is not "biasvec" - invalid format']);
end
obsize = fscanf(fid, '%d', 1);
if obsize ~= O
  error(['output bias size of ',num2str(obsize),' is not O(',num2str(O),')']);
end
mlp.OB = fscanf(fid, '%f', obsize);

fclose(fid);

