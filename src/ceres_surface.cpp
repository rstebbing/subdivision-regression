// ceres_surface.cpp

// Includes
#include "ceres_surface.h"

#include <algorithm>
#include "ceres/internal/fixed_array.h"

// SurfaceCostFunction
SurfaceCostFunction::SurfaceCostFunction(const Surface* surface)
    : surface_(surface) {
  mutable_parameter_block_sizes()->push_back(2);
  for (int i = 0; i < surface_->number_of_vertices(); ++i) {
    mutable_parameter_block_sizes()->push_back(3);
  }

  set_num_residuals(3);
}

bool SurfaceCostFunction::Evaluate(const double* const* x,
                                   double* e, double** J) const {
  // `u` and `alpha` are taken from input parameters `x`.
  double u[2] = {x[0][0], x[0][1]};
  const int p = DecodePatchIndexInPlace(u);
  CHECK_GE(p, 0);
  CHECK_LT(p, surface_->number_of_patches());
  CHECK_GE(u[0], 0.0);
  CHECK_LE(u[0], 1.0);
  CHECK_GE(u[1], 0.0);
  CHECK_LE(u[1], 1.0);

  // Get the patch vertex indices.
  auto& patch_vertex_indices = surface_->patch_vertex_indices(p);
  const int num_patch_vertices = static_cast<int>(patch_vertex_indices.size());

  // Construct `Xp`; the array of pointers to the patch vertices.
  ceres::internal::FixedArray<const double*> Xp(num_patch_vertices);
  for (int i = 0; i < num_patch_vertices; ++i) {
    Xp[i] = x[1 + patch_vertex_indices[i]];
  }

  // Can exit early when no Jacobian is required.
  if (J == nullptr) {
    return EvaluateImpl(p, u, Xp.get(), e, nullptr);
  }

  // Set Jacobians for *all* vertices (initially) to zero.
  for (int i = 0; i < surface_->number_of_vertices(); ++i) {
    if (J[1 + i] != nullptr) {
      std::fill(J[1 + i], J[1 + i] + num_residuals() * 3, 0.0);
    }
  }

  // Construct `Jp`; the array of pointers to the Jacobians.
  ceres::internal::FixedArray<double*> Jp(1 + num_patch_vertices);
  Jp[0] = J[0];
  for (int i = 0; i < num_patch_vertices; ++i) {
    Jp[i + 1] = J[1 + patch_vertex_indices[i]];
  }

  return EvaluateImpl(p, u, Xp.get(), e, Jp.get());
}

// SurfacePositionCostFunction
SurfacePositionCostFunction::SurfacePositionCostFunction(
  const Surface* surface)
    : SurfaceCostFunction(surface)
  {}

bool SurfacePositionCostFunction::EvaluateImpl(const int p,
                                               const double* u,
                                               const double* const* X,
                                               double* r,
                                               double** J) const {
  surface_->M(r, p, u, X);

  if (J != nullptr) {
    if (J[0] != nullptr) {
      double Ju[3], Jv[3];
      surface_->Mu(Ju, p, u, X);
      surface_->Mv(Jv, p, u, X);
      J[0][0] = Ju[0]; J[0][1] = Jv[0];
      J[0][2] = Ju[1]; J[0][3] = Jv[1];
      J[0][4] = Ju[2]; J[0][5] = Jv[2];
    }

    const int num_patch_vertices = static_cast<int>(
      surface_->patch_vertex_indices(p).size());
    ceres::internal::FixedArray<double> Mx(3 * 3 * num_patch_vertices);
    double* Mx_data = Mx.begin();
    surface_->Mx(Mx_data, p, u, X);

    for (int i = 0; i < num_patch_vertices; ++i) {
      if (J[i + 1] != nullptr) {
        std::copy(Mx_data + 9 * i, Mx_data + 9 * (i + 1), J[i + 1]);
      }
    }
  }

  return true;
}
