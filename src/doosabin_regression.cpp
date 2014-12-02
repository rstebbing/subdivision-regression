// doosabin_regression.cpp

// Includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include <Eigen/Dense>

#include "Math/linalg.h"
#include "Ceres/compose_cost_functions.h"

#include "doosabin.h"

#include "surface.h"
#include "ceres_surface.h"

#include "doosabin_regression.pb.h"

// DooSabinSurface
class DooSabinSurface : public Surface {
 public:
  typedef doosabin::Surface<double> Surface;

  explicit DooSabinSurface(const Surface* surface)
    : surface_(surface)
  {}

  #define EVALUATE(M) \
  virtual void M(double* r, const int p, const double* u, \
                 const double* const* X) const { \
    const Eigen::Map<const Eigen::Vector2d> _u(u); \
    const linalg::MatrixOfColumnPointers<double> _X( \
      X, 3, surface_->patch_vertex_indices(p).size()); \
    Eigen::Map<Eigen::Vector3d> _r(r); \
    surface_->M(p, _u, _X, &_r); \
  }
  EVALUATE(M);
  EVALUATE(Mu);
  EVALUATE(Mv);
  EVALUATE(Muu);
  EVALUATE(Muv);
  EVALUATE(Mvv);
  #undef EVALUATE

  // `X` is only used by `surface_` to infer the geometry dimensions.
  #define EVALUATE_X(M) \
  virtual void M(double* r, const int p, const double* u) const { \
    const Eigen::Map<const Eigen::Vector2d> _u(u); \
    const Eigen::Matrix<double, 3, 0> X; \
    Eigen::Map<Eigen::VectorXd> _r( \
      r, 3 * 3 * surface_->patch_vertex_indices(p).size()); \
    surface_->M(p, _u, X, &_r); \
  }
  EVALUATE_X(Mx);
  EVALUATE_X(Mux);
  EVALUATE_X(Mvx);
  #undef EVALUATE_X

  virtual int number_of_vertices() const {
    return surface_->number_of_vertices();
  }

  virtual int number_of_faces() const {
    return surface_->number_of_faces();
  }

  virtual int number_of_patches() const {
    return surface_->number_of_patches();
  }

  virtual const std::vector<int>& patch_vertex_indices(const int p) const {
    return surface_->patch_vertex_indices(p);
  }

  virtual const std::vector<int>& adjacent_patch_indices(const int p) const {
    return surface_->adjacent_patch_indices(p);
  }

 private:
  const Surface* surface_;
};

// LoadProblemFromFile
bool LoadProblemFromFile(const std::string& input_path,
                         Eigen::MatrixXd* Y,
                         std::vector<int>* raw_face_array,
                         Eigen::MatrixXd* X,
                         Eigen::VectorXi* p,
                         Eigen::MatrixXd* U) {
  doosabin_regression::Problem problem;
  {
    std::fstream input(input_path, std::ios::in | std::ios::binary);
    if (!input) {
      LOG(ERROR) << "File \"" << input_path << "\" not found.";
      return false;
    } else if (!problem.ParseFromIstream(&input)) {
      LOG(ERROR) << "Failed to parse input problem.";
      return false;
    }
  }

  #define LOAD_MATRIX(SRC, DST, DIM) { \
    auto& _SRC = problem.SRC(); \
    CHECK_EQ(_SRC.size() % DIM, 0); \
    DST->resize(DIM, _SRC.size() / DIM); \
    std::copy(_SRC.data(), _SRC.data() + _SRC.size(), DST->data()); \
  }

  #define LOAD_VECTOR(SRC, DST) { \
    auto& _SRC = problem.SRC(); \
    DST->resize(_SRC.size()); \
    std::copy(_SRC.data(), _SRC.data() + _SRC.size(), DST->data()); \
  }

  LOAD_MATRIX(y_3, Y, 3);
  LOAD_VECTOR(t, raw_face_array);
  LOAD_MATRIX(x_3, X, 3);
  LOAD_VECTOR(p, p);
  LOAD_MATRIX(u_2, U, 2);

  #undef LOAD_MATRIX
  #undef LOAD_VECTOR

  return true;
}

// UpdateProblemToFile
bool UpdateProblemToFile(const std::string& input_path,
                         const std::string& output_path,
                         const Eigen::MatrixXd& X,
                         const Eigen::VectorXi& p,
                         const Eigen::MatrixXd& U) {
  doosabin_regression::Problem problem;
  {
    std::fstream input(input_path, std::ios::in | std::ios::binary);
    if (!input) {
      LOG(ERROR) << "File \"" << input_path << "\" not found.";
      return false;
    } else if (!problem.ParseFromIstream(&input)) {
      LOG(ERROR) << "Failed to parse input problem.";
      return false;
    }
  }

  std::copy(X.data(), X.data() + X.rows() * X.cols(),
            problem.mutable_x_3()->begin());
  std::copy(p.data(), p.data() + p.size(),
            problem.mutable_p()->begin());
  std::copy(U.data(), U.data() + U.rows() * U.cols(),
            problem.mutable_u_2()->begin());

  {
    std::fstream output(output_path, std::ios::out |
                                     std::ios::trunc |
                                     std::ios::binary);
    if (!problem.SerializeToOstream(&output)) {
      LOG(ERROR) << "Failed to write to \"" << output_path << "\"";
      return false;
    }
  }

  return true;
}

// PositionErrorFunctor
class PositionErrorFunctor {
 public:
  template <typename M>
  PositionErrorFunctor(const M& m0, const double& sqrt_w = 1.0)
    : m0_(m0), sqrt_w_(sqrt_w)
  {}

  template <typename T>
  bool operator()(const T* m, T* e) const {
    e[0] = T(sqrt_w_) * (T(m0_[0]) - m[0]);
    e[1] = T(sqrt_w_) * (T(m0_[1]) - m[1]);
    e[2] = T(sqrt_w_) * (T(m0_[2]) - m[2]);
    return true;
  }

 private:
  Eigen::Vector3d m0_;
  const double sqrt_w_;
};

// PairwiseErrorFunctor
class PairwiseErrorFunctor {
 public:
  PairwiseErrorFunctor(const double& sqrt_w = 1.0)
    : sqrt_w_(sqrt_w)
  {}

  template <typename T>
  bool operator()(const T* x0, const T* x1, T* e) const {
    e[0] = T(sqrt_w_) * (x1[0] - x0[0]);
    e[1] = T(sqrt_w_) * (x1[1] - x0[1]);
    e[2] = T(sqrt_w_) * (x1[2] - x0[2]);
    return true;
  }

 private:
  const double sqrt_w_;
};

// main
DEFINE_int32(max_num_iterations, 1000, "Maximum number of iterations.");
DEFINE_double(function_tolerance, 0.0,
  "Minimizer terminates when "
  "(new_cost - old_cost) < function_tolerance * old_cost");
DEFINE_double(gradient_tolerance, 0.0,
  "Minimizer terminates when "
  "max_i |gradient_i| < gradient_tolerance * max_i|initial_gradient_i|");
DEFINE_double(parameter_tolerance, 0.0,
  "Minimizer terminates when "
  "|step|_2 <= parameter_tolerance * ( |x|_2 +  parameter_tolerance)");
DEFINE_double(min_trust_region_radius, 1e-9,
  "Minimizer terminates when the trust region radius becomes smaller than "
  "this value");

DEFINE_int32(num_threads, 1, "Number of threads to use for Jacobian evaluation.");
DEFINE_int32(num_linear_solver_threads, 1, "Number of threads to use for the linear solver.");

int main(int argc, char** argv) {
 GOOGLE_PROTOBUF_VERIFY_VERSION;

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (argc < 4) {
    LOG(ERROR) << "Usage: " << argv[0] << " input_path lambda output_path";
    return -1;
  }

  // Load the problem data ...
  Eigen::MatrixXd Y;
  std::vector<int> raw_face_array;
  Eigen::MatrixXd X;
  Eigen::VectorXi p;
  Eigen::MatrixXd U;
  if (!LoadProblemFromFile(argv[1], &Y, &raw_face_array, &X, &p, &U)) {
    return -1;
  }

  // ... and check consistency of dimensions.
  typedef Eigen::DenseIndex Index;
  const Index num_data_points = Y.cols();
  CHECK_EQ(num_data_points, p.size());
  CHECK_EQ(num_data_points, U.cols());

  doosabin::GeneralMesh T(std::move(raw_face_array));
  doosabin::Surface<double> surface(T);
  DooSabinSurface doosabin_surface(&surface);
  CHECK_EQ(surface.number_of_vertices(), X.cols());

  // Set lambda.
  const double lambda = atof(argv[2]);

  // Setup `problem`.
  ceres::Problem problem;

  // Encode patch indices into the preimage positions.
  for (Index i = 0; i < num_data_points; ++i) {
    EncodePatchIndexInPlace(U.data() + 2 * i, p[i]);
  }

  // Add error residuals.
  std::vector<double*> parameter_blocks;
  parameter_blocks.reserve(1 + surface.number_of_vertices());
  parameter_blocks.push_back(nullptr); // `u`.
  for (Index i = 0; i < X.cols(); ++i) {
    parameter_blocks.push_back(X.data() + 3 * i);
  }

  for (Index i = 0; i < num_data_points; ++i) {
    parameter_blocks[0] = U.data() + 2 * i;

    auto position_error = new ceres_utility::GeneralCostFunctionComposition(
      new ceres::AutoDiffCostFunction<PositionErrorFunctor, 3, 3>(
        new PositionErrorFunctor(Y.col(i), 1.0 / sqrt(num_data_points))));
    position_error->AddInputCostFunction(
      new SurfacePositionCostFunction(&doosabin_surface),
      parameter_blocks);
    position_error->Finalise();

    problem.AddResidualBlock(position_error, nullptr, parameter_blocks);
  }

  // Add regularisation residuals.
  std::set<std::pair<int, int>> edges;
  for (auto& half_edge : T.iterate_half_edges()) {
    int i = half_edge.first, j = half_edge.second;
    if (i > j) {
      std::swap(i, j);
    }
    auto full_edge = std::make_pair(i, j);
    if (edges.count(full_edge)) {
      continue;
    }

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PairwiseErrorFunctor, 3, 3, 3>(
      new PairwiseErrorFunctor(sqrt(lambda))),
      nullptr,
      X.data() + 3 * i,
      X.data() + 3 * j);

    edges.insert(full_edge);
  }

  typedef doosabin::SurfaceWalker<double> DooSabinWalker;
  DooSabinWalker walker(&surface);
  for (Index i = 0; i < num_data_points; ++i) {
    problem.SetParameterization(
      U.data() + 2 * i,
      new PreimageLocalParameterisation<DooSabinWalker, Eigen::MatrixXd>(
        &walker, &X));
  }

  // Initialise the solver options.
  std::cout << "Solver options:" << std::endl;
  ceres::Solver::Options options;

  options.num_threads = FLAGS_num_threads;
  options.num_linear_solver_threads = FLAGS_num_linear_solver_threads;

  // Disable auto-scaling and set LM to be additive (instead of multiplicative).
  options.min_lm_diagonal = 1.0;
  options.max_lm_diagonal = 1.0;
  options.jacobi_scaling = false;

  options.minimizer_progress_to_stdout = FLAGS_v >= 1;

  // Termination criteria.
  options.max_num_iterations = FLAGS_max_num_iterations;
  std::cout << " max_num_iterations: " << options.max_num_iterations <<
               std::endl;
  options.max_num_consecutive_invalid_steps = FLAGS_max_num_iterations;

  options.function_tolerance = FLAGS_function_tolerance;
  options.gradient_tolerance = FLAGS_gradient_tolerance;
  options.parameter_tolerance = FLAGS_parameter_tolerance;
  options.min_trust_region_radius = FLAGS_min_trust_region_radius;

  // Solver selection.
  options.dynamic_sparsity = true;

  // `update_state_every_iteration` is required by
  // `PreimageLocalParameterisation` instances.
  options.update_state_every_iteration = true;

  // Solve.
  std::cout << "Solving ..." << std::endl;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;

  // Decode patch indices.
  for (Index i = 0; i < num_data_points; ++i) {
    p[i] = DecodePatchIndexInPlace(U.data() + 2 * i);
  }

  // Save.
  if (!UpdateProblemToFile(argv[1], argv[3], X, p, U)) {
    return -1;
  }
  std::cout << "Output: " << argv[3] << std::endl;

  return 0;
}
