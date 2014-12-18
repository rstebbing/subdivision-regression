// doosabin_regression.cpp

// Includes
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <utility>

#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include <Eigen/Dense>

#include "rapidjson/document.h"
#include "rapidjson/filestream.h"
#include "rapidjson/prettywriter.h"

#include "Math/linalg.h"
#include "Ceres/compose_cost_functions.h"

#include "doosabin.h"

#include "ceres_surface.h"
#include "surface.h"

// DooSabinSurface
class DooSabinSurface : public Surface {
 public:
  typedef doosabin::Surface<double> Surface;

  explicit DooSabinSurface(const Surface* surface)
    : surface_(surface)
  {}

  #define EVALUATE(M, SIZE) \
  virtual void M(double* r, const int p, const double* u, \
                 const double* const* X) const { \
    const Eigen::Map<const Eigen::Vector2d> _u(u); \
    const linalg::MatrixOfColumnPointers<double> _X( \
      X, 3, surface_->patch_vertex_indices(p).size()); \
    Eigen::Map<Eigen::VectorXd> _r(r, SIZE); \
    surface_->M(p, _u, _X, &_r); \
  }
  #define SIZE_JACOBIAN_X (3 * 3 * surface_->patch_vertex_indices(p).size())
  EVALUATE(M, 3);
  EVALUATE(Mu, 3);
  EVALUATE(Mv, 3);
  EVALUATE(Muu, 3);
  EVALUATE(Muv, 3);
  EVALUATE(Mvv, 3);
  EVALUATE(Mx, SIZE_JACOBIAN_X);
  EVALUATE(Mux, SIZE_JACOBIAN_X);
  EVALUATE(Mvx, SIZE_JACOBIAN_X);
  #undef EVALUATE
  #undef SIZE_JACOBIAN_X

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

// ReadFileIntoDocument
bool ReadFileIntoDocument(const std::string& input_path,
                          rapidjson::Document* document) {
  std::ifstream input(input_path, std::ios::binary);
  if (!input) {
    LOG(ERROR) << "File \"" << input_path << "\" not found.";
    return false;
  }

  std::stringstream input_ss;
  input_ss << input.rdbuf();
  if (document->Parse<0>(input_ss.str().c_str()).HasParseError()) {
    LOG(ERROR) << "Failed to parse input.";
    return false;
  }

  return true;
}

// LoadProblemFromFile
bool LoadProblemFromFile(const std::string& input_path,
                         Eigen::MatrixXd* Y,
                         std::vector<int>* raw_face_array,
                         Eigen::MatrixXd* X,
                         Eigen::VectorXi* p,
                         Eigen::MatrixXd* U) {
  rapidjson::Document document;
  if (!ReadFileIntoDocument(input_path, &document)) {
    return false;
  }

  #define LOAD_MATRIXD(SRC, DST, DIM) { \
    CHECK(document.HasMember(#SRC)); \
    auto& v = document[#SRC]; \
    CHECK(v.IsArray()); \
    DST->resize(DIM, v.Size() / DIM); \
    for (rapidjson::SizeType i = 0; i < v.Size(); ++i) { \
      CHECK(v[i].IsDouble()); \
      (*DST)(i % DIM, i / DIM) = v[i].GetDouble(); \
    } \
  }

  #define LOAD_VECTORI(SRC, DST) { \
    CHECK(document.HasMember(#SRC)); \
    auto& v = document[#SRC]; \
    CHECK(v.IsArray()); \
    DST->resize(v.Size()); \
    for (rapidjson::SizeType i = 0; i < v.Size(); ++i) { \
      CHECK(v[i].IsInt()); \
      (*DST)[i] = v[i].GetInt(); \
    } \
  }

  LOAD_MATRIXD(Y, Y, 3);
  LOAD_VECTORI(raw_face_array, raw_face_array);
  LOAD_MATRIXD(X, X, 3);
  LOAD_VECTORI(p, p);
  LOAD_MATRIXD(U, U, 2);

  #undef LOAD_MATRIXD
  #undef LOAD_VECTORI

  return true;
}

// UpdateProblemToFile
bool UpdateProblemToFile(const std::string& input_path,
                         const std::string& output_path,
                         const Eigen::MatrixXd& X,
                         const Eigen::VectorXi& p,
                         const Eigen::MatrixXd& U) {

  rapidjson::Document document;
  if (!ReadFileIntoDocument(input_path, &document)) {
    return false;
  }

  #define SAVE_MATRIXD(SRC, DST) { \
    CHECK(document.HasMember(#DST)); \
    auto& v = document[#DST]; \
    CHECK(v.IsArray()); \
    CHECK_EQ(v.Size(), SRC.rows() * SRC.cols()); \
    for (rapidjson::SizeType i = 0; i < v.Size(); ++i) { \
      CHECK(v[i].IsDouble()); \
      v[i] = SRC(i % SRC.rows(), i / SRC.rows()); \
    } \
  }

  #define SAVE_VECTORI(SRC, DST) { \
    CHECK(document.HasMember(#DST)); \
    auto& v = document[#DST]; \
    CHECK(v.IsArray()); \
    CHECK_EQ(v.Size(), SRC.size()); \
    for (rapidjson::SizeType i = 0; i < v.Size(); ++i) { \
      CHECK(v[i].IsInt()); \
      v[i] = SRC[i]; \
    } \
  }

  SAVE_MATRIXD(X, X);
  SAVE_VECTORI(p, p);
  SAVE_MATRIXD(U, U);

  // Use `fopen` instead of streams for `rapidjson::FileStream`.
  FILE* output_handle = fopen(output_path.c_str(), "wb");
  if (output_handle == nullptr) {
    LOG(ERROR) << "Unable to open \"" << output_path << "\"";
  }

  rapidjson::FileStream output(output_handle);
  rapidjson::PrettyWriter<rapidjson::FileStream> writer(output);
  document.Accept(writer);

  fclose(output_handle);

  return true;
}

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

DEFINE_int32(num_threads, 1,
  "Number of threads to use for Jacobian evaluation.");
DEFINE_int32(num_linear_solver_threads, 1,
  "Number of threads to use for the linear solver.");

int main(int argc, char** argv) {
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
  CHECK_GT(num_data_points, 0);
  CHECK_EQ(num_data_points, p.size());
  CHECK_EQ(num_data_points, U.cols());

  doosabin::GeneralMesh T(std::move(raw_face_array));
  CHECK_GT(T.number_of_faces(), 0);
  doosabin::Surface<double> surface(T);
  DooSabinSurface doosabin_surface(&surface);
  CHECK_EQ(surface.number_of_vertices(), X.cols());

  // `lambda` is the regularisation weight.
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

  std::unique_ptr<ceres::CostFunction> surface_position(
    new SurfacePositionCostFunction(&doosabin_surface));

  for (Index i = 0; i < num_data_points; ++i) {
    parameter_blocks[0] = U.data() + 2 * i;

    auto position_error = new ceres_utility::GeneralCostFunctionComposition(
      new ceres::AutoDiffCostFunction<PositionErrorFunctor, 3, 3>(
        new PositionErrorFunctor(Y.col(i), 1.0 / sqrt(num_data_points))));
    position_error->AddInputCostFunction(surface_position.get(),
                                         parameter_blocks,
                                         false);
    position_error->Finalise();

    problem.AddResidualBlock(position_error, nullptr, parameter_blocks);
  }

  // Add regularisation residuals.
  // Note: `problem` takes ownership of `pairwise_error`.
  auto pairwise_error =
    new ceres::AutoDiffCostFunction<PairwiseErrorFunctor, 3, 3, 3>(
      new PairwiseErrorFunctor(sqrt(lambda)));

  std::set<std::pair<int, int>> full_edges;
  for (std::pair<int, int> e : T.iterate_half_edges()) {
    if (e.first > e.second) {
      std::swap(e.first, e.second);
    }
    if (!full_edges.count(e)) {
      problem.AddResidualBlock(pairwise_error,
                               nullptr,
                               X.data() + 3 * e.first,
                               X.data() + 3 * e.second);
      full_edges.insert(e);
    }
  }

  // Set preimage parameterisations so that they are updated using
  // `doosabin::SurfaceWalker<double>` and NOT Euclidean addition.
  // Note: `problem` takes ownership of `local_parameterisation`.
  typedef doosabin::SurfaceWalker<double> DooSabinWalker;
  DooSabinWalker walker(&surface);
  auto local_parameterisation =
    new PreimageLocalParameterisation<DooSabinWalker, Eigen::MatrixXd>(
      &walker, &X);

  for (Index i = 0; i < num_data_points; ++i) {
    problem.SetParameterization(U.data() + 2 * i, local_parameterisation);
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
