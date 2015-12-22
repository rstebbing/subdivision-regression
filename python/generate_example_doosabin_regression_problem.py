#########################################################
# File: generate_example_doosabin_regression_problem.py #
# Copyright Richard Stebbing 2015.                      #
# Distributed under the MIT License.                    #
# (See accompany file LICENSE or copy at                #
#  http://opensource.org/licenses/MIT)                  #
#########################################################

# Imports
import argparse
import numpy as np
import json

from sklearn.neighbors import NearestNeighbors

# Requires `subdivision`.
from subdivision import doosabin

# Requires `rscommon`.
from rscommon.face_array import sequence_to_raw_face_array

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_data_points', type=int)
    parser.add_argument('output_path')
    parser.add_argument('--radius-std-dev', type=float, default=0.05)
    parser.add_argument('--hemisphere', default=False, action='store_true')
    parser.add_argument('--num-subdivisions', type=int, default=0)
    parser.add_argument('--initialisation-sample-density', type=int,
                        default=16)
    args = parser.parse_args()

    # Generate matrix `Y` of points uniformly sampled on a sphere then
    # displaced radially with zero-mean Gaussian noise.
    Y = np.random.randn(args.num_data_points * 3).reshape(-1, 3)
    n = np.linalg.norm(Y, axis=1)
    n[n <= 0.0] = 1.0
    r = (1.0 + args.radius_std_dev * np.random.randn(args.num_data_points))
    Y *= (r / n)[:, np.newaxis]
    if args.hemisphere:
        Y[Y[:, 2] < 0.0, 2] *= -1.0

    # Initial mesh and geometry is the cube from:
    # subdivision/doosabin/examples/visualise_doosabin_subdivision.py.
    T = [[0, 1, 3, 2],
         [4, 6, 7, 5],
         [1, 5, 7, 3],
         [6, 4, 0, 2],
         [0, 4, 5, 1],
         [3, 7, 6, 2]]
    X = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]], dtype=np.float64)
    X -= np.mean(X, axis=0)

    num_subdivisions = max(args.num_subdivisions,
                           doosabin.is_initial_subdivision_required(T))
    for i in xrange(num_subdivisions):
        T, X = doosabin.subdivide(T, X)

    # Initialise preimages.
    surface = doosabin.surface(T)
    pd, Ud, Td = surface.uniform_parameterisation(
        args.initialisation_sample_density)
    M = surface.M(pd, Ud, X)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(M)
    i = nn.kneighbors(Y, return_distance=False).ravel()
    p, U = pd[i], Ud[i]

    # Output.
    z = {}
    def z_setitem(k, A):
        z[k] = np.asarray(A).ravel().tolist()

    z_setitem('Y', Y)
    z_setitem('raw_face_array', sequence_to_raw_face_array(T))
    z_setitem('X', X)
    z_setitem('p', p)
    z_setitem('U', U)

    print 'Output:', args.output_path
    with open(args.output_path, 'wb') as fp:
        json.dump(z, fp, indent=4)

if __name__ == '__main__':
    main()
