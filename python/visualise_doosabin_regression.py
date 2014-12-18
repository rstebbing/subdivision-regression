# visualise_doosabin_regression.py

# Imports
import argparse
import numpy as np
import json

# Requires `subdivision`.
from subdivision import doosabin

# Requires `rscommon`.
from rscommon.face_array import raw_face_array_to_sequence
from rscommon import vtk_

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('-d', '--disable', action='append', default=[],
                        choices={'Y', 'T', 'M', 'U'})
    parser.add_argument('--opacity', type=float, default=0.6)
    parser.add_argument('--sample-density', type=int, default=16)
    args = parser.parse_args()

    with open(args.input_path, 'rb') as fp:
        z = json.loads(fp.read())
    def z_getitem(k, cols=None):
        a = np.array(z[k])
        return a.reshape(-1, cols) if cols is not None else a

    Y = z_getitem('Y', 3)
    T = raw_face_array_to_sequence(z_getitem('raw_face_array'))
    X = z_getitem('X', 3)
    p = z_getitem('p')
    U = z_getitem('U', 2)

    surface = doosabin.surface(T)
    pd, Ud, Td = surface.uniform_parameterisation(args.sample_density)
    M = surface.M(pd, Ud, X)

    a = {}
    if 'Y' not in args.disable:
        a['Y'] = vtk_.points(Y, ('SetRadius', 0.025),
                                ('SetThetaResolution', 16),
                                ('SetPhiResolution', 16))

    camera = vtk_.vtk.vtkCamera()
    if 'M' not in args.disable:
        a['M'] = vtk_.surface(Td, M, camera=camera)
        a['M'].GetProperty().SetColor(0.216, 0.494, 0.722)
        a['M'].GetProperty().SetSpecular(1.0)
        a['M'].GetProperty().SetOpacity(args.opacity)

    MU = surface.M(p, U, X)
    if 'U' not in args.disable:
        a['MU'] = vtk_.points(MU, ('SetRadius', 0.015),
                                  ('SetThetaResolution', 16),
                                  ('SetPhiResolution', 16))

        i = np.arange(p.size)
        a['U'] = vtk_.tubes(np.c_[i, i + p.size], np.r_['0,2', MU, Y],
                            ('SetRadius', 0.01),
                            ('SetNumberOfSides', 16))

    if 'T' not in args.disable:
        a['T_points'] = vtk_.points(X, ('SetRadius', 0.05),
                                       ('SetThetaResolution', 16),
                                       ('SetPhiResolution', 16))
        a['T_mesh'] = vtk_.mesh(T, X, ('SetRadius', 0.025),
                                      ('SetNumberOfSides', 16))

    ren, iren = vtk_.renderer(*a.values())

    ren.SetActiveCamera(camera)
    ren.ResetCamera()

    iren.Initialize()
    iren.Start()

if __name__ == '__main__':
    main()
