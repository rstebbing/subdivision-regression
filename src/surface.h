// surface.h
#ifndef SURFACE_H
#define SURFACE_H

// Includes
#include <vector>

// Surface
struct Surface {
  virtual void M(double* m, const int p, const double* u, const double* const* X) const = 0;
  virtual void Mu(double* mu, const int p, const double* u, const double* const* X) const = 0;
  virtual void Mv(double* mv, const int p, const double* u, const double* const* X) const = 0;
  virtual void Muu(double* muu, const int p, const double* u, const double* const* X) const = 0;
  virtual void Muv(double* muv, const int p, const double* u, const double* const* X) const = 0;
  virtual void Mvv(double* mvv, const int p, const double* u, const double* const* X) const = 0;
  virtual void Mx(double* mx, const int p, const double* u) const = 0;
  virtual void Mux(double* mux, const int p, const double* u) const = 0;
  virtual void Mvx(double* mvx, const int p, const double* u) const = 0;

  virtual int number_of_vertices() const = 0;
  virtual int number_of_faces() const = 0;
  virtual int number_of_patches() const = 0;
  virtual const std::vector<int>& patch_vertex_indices(const int p) const = 0;
  virtual const std::vector<int>& adjacent_patch_indices(const int p) const = 0;
};

#endif // SURFACE_H
