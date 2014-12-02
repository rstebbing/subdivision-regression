// ceres_surface.h
#ifndef CERES_SURFACE_H
#define CERES_SURFACE_H

// Includes
#include "ceres/ceres.h"

#include "patch_index_encoding.h"
#include "surface.h"

// PreimageLocalParameterisation
template <typename Walker, typename TX>
class PreimageLocalParameterisation :
  public ceres::LocalParameterization {
 public:
  PreimageLocalParameterisation(const Walker* walker, const TX* X)
      : walker_(walker), X_(X) {}

  virtual bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const {
    double u[2] = {x[0], x[1]};
    int p = DecodePatchIndexInPlace(u);
    int p1 = -1;
    walker_->ApplyUpdate(*X_, p, u, delta, &p1, &x_plus_delta);
    CHECK_GE(p1, 0);
    EncodePatchIndexInPlace(x_plus_delta, p1);
    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* J) const {
    J[0] = 1.0; J[1] = 0.0;
    J[2] = 0.0; J[3] = 1.0;
    return true;
  }

  virtual int GlobalSize() const { return 2; }
  virtual int LocalSize() const { return 2; }

 private:
  const Walker* walker_;
  const TX* X_;
};

// SurfaceCostFunction
class SurfaceCostFunction : public ceres::CostFunction {
 public:
  SurfaceCostFunction(
    const Surface* surface,
    const int D = 1);

  virtual bool Evaluate(const double* const* x, double* e, double** J) const;

 protected:
  virtual bool EvaluateImpl(const int p,
                            const double* u,
                            const double* const* X,
                            double* r,
                            double** J) const = 0;

protected:
  const Surface* surface_;
  const int D_;
};

// SurfacePositionCostFunction
class SurfacePositionCostFunction : public SurfaceCostFunction {
 public:
  SurfacePositionCostFunction(
    const Surface* surface,
    const int D = 1);

 protected:
  virtual bool EvaluateImpl(const int p,
                            const double* u,
                            const double* const* X,
                            double* r,
                            double** J) const;
};

#endif // CERES_SURFACE_H
