// ceres_surface.cpp

// Includes
#include "ceres_surface.h"

#include <algorithm>
#include "ceres/internal/fixed_array.h"

// SurfaceCostFunction
SurfaceCostFunction::SurfaceCostFunction(
  const Surface* surface,
  const int D)
  : surface_(surface),
    D_(D) {
  const int kDim = 3;

  mutable_parameter_block_sizes()->push_back(2);
  if (D_ > 1)
    mutable_parameter_block_sizes()->push_back(D_ - 1);
  for (int i = 0; i < D_; ++i) {
    for (int j = 0; j < surface_->number_of_vertices(); ++j) {
      mutable_parameter_block_sizes()->push_back(kDim);
    }
  }

  set_num_residuals(kDim);
}

bool SurfaceCostFunction::Evaluate(
  const double* const* x, double* e, double** J) const {
  const int kDim = 3;

  // `u` and `alpha` are taken from input parameters `x`.
  double u[2] = {x[0][0], x[0][1]};
  const int p = DecodePatchIndexInPlace(u);
  CHECK_GE(p, 0);
  CHECK_LT(p, surface_->number_of_patches());
  CHECK_GE(u[0], 0.0);
  CHECK_LE(u[0], 1.0);
  CHECK_GE(u[1], 0.0);
  CHECK_LE(u[1], 1.0);

  const double* alpha = D_ > 1 ? x[1] : nullptr;

  // Get the patch vertex indices.
  auto& patch_vertex_indices = surface_->patch_vertex_indices(p);
  const int num_patch_vertices = (int)patch_vertex_indices.size();

  // Construct `Xp_data`; the blended patch vertices.
  ceres::internal::FixedArray<double> Xp_data(num_patch_vertices * kDim);

  const int kXOffset = D_ > 1 ? 2 : 1;
  for (int i = 0; i < num_patch_vertices; ++i) {
    for (int j = 0; j < kDim; ++j)
      Xp_data[kDim * i + j] = x[kXOffset + patch_vertex_indices[i]][j];
  }

  for (int m = 1; m < D_; ++m) {
    for (int i = 0; i < num_patch_vertices; ++i) {
      for (int j = 0; j < kDim; ++j)
        Xp_data[kDim * i + j] +=
          alpha[m - 1] * x[kXOffset +
                           m * surface_->number_of_vertices() +
                           patch_vertex_indices[i]][j];
    }
  }

  // Construct `Xp`; the array of pointers to the blended patch vertices.
  ceres::internal::FixedArray<const double*> Xp(num_patch_vertices);
  for (int i = 0; i < num_patch_vertices; ++i)
    Xp[i] = &Xp_data[kDim * i];

  // Can exit early when no Jacobian is required.
  if (J == nullptr)
    return EvaluateImpl(p, u, Xp.get(), e, nullptr);

  // Otherwise, construct Jacobian data, but write Jacobian for `u`
  // directly into the output Jacobian.
  ceres::internal::FixedArray<double> Jp_data(
    num_residuals() * kDim * num_patch_vertices);
  ceres::internal::FixedArray<double*> Jp(1 + num_patch_vertices);
  Jp[0] = J[0];
  for (int i = 0; i < num_patch_vertices; ++i)
    Jp[i + 1] = Jp_data.get() + num_residuals() * kDim * i;

  if (!EvaluateImpl(p, u, Xp.get(), e, Jp.get()))
    return false;

  // If single dimension then just copy the relevant entries into
  // the output.
  if (D_ == 1) {
    DCHECK_EQ(kXOffset, 1);
    for (int i = 0; i < surface_->number_of_vertices(); ++i) {
      if (J[kXOffset + i] != nullptr) {
        std::fill(J[kXOffset + i],
                  J[kXOffset + i] + num_residuals() * kDim,
                  0.0);
      }
    }

    for (int i = 0; i < num_patch_vertices; ++i) {
      if (J[kXOffset + patch_vertex_indices[i]] == nullptr)
          continue;

      std::copy(Jp[i + 1],
                Jp[i + 1] + num_residuals() * kDim,
                J[kXOffset + patch_vertex_indices[i]]);
    }
    return true;
  }

  DCHECK_GE(D_, 2);

  // Apply the chain rule to set Jacobians for `alpha` and `X`.
  DCHECK_EQ(kXOffset, 2);
  if (J[1] != nullptr) {
    for (int m = 0; m < num_residuals(); ++m) {
      for (int n = 1; n < D_; ++n) {
        J[1][m * (D_ - 1) + n - 1] = 0.0;
        for (int i = 0; i < num_patch_vertices; ++i)
          for (int k = 0; k < kDim; ++k)
            J[1][m * (D_ - 1) + n - 1] += Jp[i + 1][kDim * m + k] *
              x[kXOffset + n * surface_->number_of_vertices() +
                patch_vertex_indices[i]][k];
      }
    }
  }

  for (int m = 0; m < D_; ++m) {
    for (int i = 0; i < surface_->number_of_vertices(); ++i) {
      if (J[kXOffset + m * surface_->number_of_vertices() + i] != nullptr) {
        std::fill(J[kXOffset + m * surface_->number_of_vertices() + i],
                  J[kXOffset + m * surface_->number_of_vertices() + i] +
                    num_residuals() * kDim,
                  0.0);
      }
    }

    const double& a = (m < 1) ? 1.0 : alpha[m - 1];

    for (int i = 0; i < num_patch_vertices; ++i) {
      for (int j = 0; j < num_residuals(); ++j) {
        for (int k = 0; k < kDim; ++k) {
          J[kXOffset + m * surface_->number_of_vertices() +
            patch_vertex_indices[i]][j * kDim + k] =
              a * Jp[i + 1][j * kDim + k];
        }
      }
    }
  }

  return true;
}

// SurfacePositionCostFunction
SurfacePositionCostFunction::SurfacePositionCostFunction(
  const Surface* surface,
  const int D)
  : SurfaceCostFunction(surface, D) {}

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

    const int num_patch_vertices = surface_->patch_vertex_indices(p).size();
    ceres::internal::FixedArray<double> Mx(3 * 3 * num_patch_vertices);
    double* Mx_data = Mx.begin();
    surface_->Mx(Mx_data, p, u);

    for (int i = 0; i < num_patch_vertices; ++i) {
      if (J[i + 1] != nullptr) {
        std::copy(Mx_data + 9 * i, Mx_data + 9 * (i + 1), J[i + 1]);
      }
    }
  }

  return true;
}
