#!/usr/local/bin/ipython --gui=wx
# Adapted from SPLOCS view_animation.py by Thomas Neumann, see https://github.com/tneumann/splocs
# Copyright (c) [2015] [Javier Romero]

from argparse import ArgumentParser
import h5py
from itertools import count
from mayavi import mlab
from tvtk.api import tvtk

def main(hdf5_animation_file, sid='50004', pid='jiggle_on_toes'):
    with h5py.File(hdf5_animation_file, 'r') as f:
        verts = f[sid+'_'+pid].value.transpose([2,0,1])
        tris = f['faces'].value

    pd = tvtk.PolyData(points=verts[0], polys=tris)
    normals = tvtk.PolyDataNormals(input=pd, splitting=False)
    actor = tvtk.Actor(mapper=tvtk.PolyDataMapper(input=normals.output))
    actor.property.set(edge_color=(0.5, 0.5, 0.5), ambient=0.0,
                       specular=0.15, specular_power=128., shading=True, diffuse=0.8)

    fig = mlab.figure(bgcolor=(1,1,1))
    fig.scene.add_actor(actor)

    @mlab.animate(delay=40, ui=False)
    def animation():
        for i in count():
            frame = i % len(verts)
            pd.points = verts[frame]
            fig.scene.render()
            yield

    a = animation()
    fig.scene.z_minus_view()
    mlab.show()

if __name__ == '__main__':

    sids = ['50004', '50020', '50021', '50022', '50025',
            '50002', '50007', '50009', '50026', '50027']
    pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot']

    parser = ArgumentParser(description='Save sequence meshes as obj')
    parser.add_argument('--path', type=str, default='./dyna_female.hdf5',
                        help='dataset path in hdf5 format')
    parser.add_argument('--seq', type=str, default='jiggle_on_toes',
                        choices=pids, help='sequence name')
    parser.add_argument('--sid', type=str, default='50004',
                        choices=sids, help='subject id')
    args = parser.parse_args()

    main(args.path, args.sid, args.seq)
