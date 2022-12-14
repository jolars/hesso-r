#pragma once

#include <Eigen/Core>
#include <vector>

namespace hesso {

template<typename Derived>
class Objective
{
public:
  Objective(const size_t p)
    : hessian_cache(p, -1.0){};

  // virtual ~Objective() {}

  double loss(const Eigen::VectorXd& residual) const
  {
    return static_cast<const Derived*>(this)->loss(residual);
  }

  double deviance(const Eigen::VectorXd& residual) const
  {
    return static_cast<const Derived*>(this)->deviance(residual);
  }

  double dual(const Eigen::VectorXd& theta, const Eigen::VectorXd& y) const
  {
    return static_cast<const Derived*>(this)->dual(theta, y);
  }

  Eigen::VectorXd residual(const Eigen::VectorXd& eta,
                           const Eigen::VectorXd& y) const
  {
    return static_cast<const Derived*>(this)->residual(eta, y);
  }

  void updateResidual(Eigen::VectorXd& residual,
                      const Eigen::VectorXd& eta,
                      const Eigen::VectorXd& y) const
  {
    return static_cast<const Derived*>(this)->updateResidual(residual, eta, y);
  }

  void updateResidual(Eigen::VectorXd& residual,
                      const double intercept_update) const
  {
    return static_cast<const Derived*>(this)->updateResidual(residual,
                                                             intercept_update);
  }

  template<typename T>
  void updateResidual(Eigen::VectorXd& residual,
                      const double beta_diff,
                      const T& x,
                      const size_t j) const
  {
    return static_cast<const Derived*>(this)->updateResidual(
      residual, beta_diff, x, j);
  }

  template<typename T>
  Eigen::MatrixXd hessian(const T& x,
                          const std::vector<size_t>& ind_a,
                          const std::vector<size_t>& ind_b) const
  {
    return static_cast<const Derived*>(this)->template hessian<T>(
      x, ind_a, ind_b);
  }

  template<typename T>
  double hessianTerm(const T& x, const size_t j)
  {
    return static_cast<Derived*>(this)->template hessianTerm<T>(x, j);
  }

protected:
  std::vector<double> hessian_cache;
};

} // namespace hesso
