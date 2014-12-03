# visualise_doosabin_regression.py

# Imports
import argparse
import numpy as np

# Requires subdivision/doosabin on `PYTHONPATH`.
import doosabin

# Requires common/python on `PYTHONPATH`.
from face_array import raw_face_array_to_sequence
import protobuf_ as pb
import vtk_

from doosabin_regression_pb2 import Problem

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('--sample-density', type=int, default=16)
    args = parser.parse_args()

    s = Problem()
    with open(args.input_path, 'rb') as fp:
        s.ParseFromString(fp.read())

    Y = pb.load_array(s, 'y', 3)
    T = raw_face_array_to_sequence(pb.load_array(s, 't'))
    X = pb.load_array(s, 'x', 3)
    p = pb.load_array(s, 'p')
    U = pb.load_array(s, 'u', 2)

    surface = doosabin.Surface(T)
    pd, Ud, Td = surface.uniform_parameterisation(args.sample_density)
    M = surface.M(pd, Ud, X)

    a = {}
    a['Y'] = vtk_.points(Y, ('SetRadius', 0.025),
                            ('SetThetaResolution', 16),
                            ('SetPhiResolution', 16))
    camera = vtk_.vtk.vtkCamera()
    a['M'] = vtk_.surface(Td, M, camera=camera)
    a['M'].GetProperty().SetColor(0.216, 0.494, 0.722)
    a['M'].GetProperty().SetSpecular(1.0)
    a['M'].GetProperty().SetOpacity(0.6)

    MU = surface.M(p, U, X)
    a['MU'] = vtk_.points(MU, ('SetRadius', 0.015),
                              ('SetThetaResolution', 16),
                              ('SetPhiResolution', 16))

    i = np.arange(p.size)
    a['U'] = vtk_.tubes(np.c_[i, i + p.size], np.r_['0,2', MU, Y],
                        ('SetRadius', 0.01),
                        ('SetNumberOfSides', 16))

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
