function visualize_mesh_sequence( verts,faces )
% VISUALIZE_MESH_SEQUENCE Visualize sequence of meshes
%   INPUTS:
%        - verts: array of size num_verts x 3 x num_frames
%        - faces: array of size num_faces x 3
%
% Copyright (c) by Gerard Pons-Moll 2015
%
close all
ha=gca;
v=verts(:,:,1);
p=patch('parent',ha,'Vertices',v,'faces',faces+1);
% Set visualization parameters
xyzmax=max(max(verts),[],3);
xyzmin=min(min(verts),[],3);
axis equal
axis([xyzmin(1) xyzmax(1) xyzmin(2) xyzmax(2) xyzmin(3) xyzmax(3)]);
%axis equal
camlight 
set(p, 'ambientStrength', 0.4);
set(p, 'diffuseStrength', 0.4);
set(p, 'FaceLighting', 'phong');
set(p, 'edgeLighting', 'phong');
set(p,'FaceColor',[0.7,0.78,1],'EdgeColor','white')
view(0,90);
set(ha, 'cameraupvector', [0 1 0]);
cameratoolbar('setmode', 'orbit');
cameratoolbar('setcoordsys', 'y');
cameratoolbar('show');

for frame=1:size(verts,3)
   set(p,'Vertices',verts(:,:,frame),'faces',faces+1)
   pause(0.1)
end
end
