function [verts,faces] = load_dyna_dataset(file,sid,seq,tdir,write_obj)
% LOAD_DYNA_DATASET
%   Loads the dyna dataset and saves the meshes of the queried sequence in *obj
%   If write_obj is set to true it saves one *obj mesh per frame in the queried
%   sequence in a subfolder in directory tdir.
%   See in training_file_<gender>.txt the available subject ids and
%   sequences.
%
%   INPUTS:
%       - file: dataset filename. (e.g., dyna_female_zip.h5)
%       - sid: queried subject id.
%       - seq: queried sequence.
%       - tdir: target directory. Parent directory where *obj meshes are
%         saved. The meshes will be saved in a subfolder with the name
%         ./sid_seq/
%       - write_obj: whether to write obj meshes. Default is false
%
%   OUTPUTS:
%        - verts: 3d array vertices of size num_verts x 3 x num_frames
%        - faces: matrix of size num_faces x 3
%
% Examples
% --------
%   
%   1) To write obj meshes
%       >> load_dyna_dataset('dyna_female.hdf5','50004','jiggle_on_toes','./',true);
%   2) To visualize a sequence:
%       >> [verts,faces]=load_dyna_dataset('dyna_male.hdf5','50002','running_on_spot','./',false);
%       >> visualize_mesh_sequence(verts,faces)
%
%   See also visualize_mesh_sequence.m
%
% Copyright (c) by Gerard Pons-Moll

if ~exist('write_obj','var')
    write_obj=false;
end
verts=h5read(file,['/',sid,'_',seq]);
verts=permute(verts,[3 2 1]);
faces=h5read(file,'/faces')';
if write_obj
    sidseq=[sid,'_',seq];
    save_dir=fullfile(tdir,sidseq);
    if ~(exist(save_dir,'dir'))
        mkdir(save_dir)
    end
    for frame=1:size(verts,3)
        v=verts(:,:,frame);
        fname=fullfile(tdir,sidseq,sprintf('%05d.obj',frame-1));
        disp(['Saving ' fname])
        write_mesh_as_obj(fname,v,faces);
    end
end
end
function write_mesh_as_obj(fname,verts,faces)
fp=fopen(fname,'w');
for kk=1:size(verts,1)
    v=verts(kk,:);
    fprintf(fp,'v %f %f %f\n', v);
end
% Faces are 1-based, not 0-based in obj files
faces=faces+1;
for kk=1:size(faces,1)
    f=faces(kk,:);
    fprintf(fp,'f %d %d %d\n',f);
end
fclose(fp);
end
