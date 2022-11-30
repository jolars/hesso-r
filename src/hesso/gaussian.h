#pragma once

#include "objective.h"
#include <Eigen/Core>

namespace hesso {

class Gaussian : public Objective<Gaussian>
{
public:
  template<typename... Ts>
  Gaussian(Ts... args)
    : Objective(std::forward<Ts>(args)...)
  {
  }

  double loss(const Eigen::VectorXd& residual) const
  {
    return 0.5 * residual.squaredNorm();
  }

  double deviance(const Eigen::VectorXd& residual) const
  {
    return residual.squaredNorm();
  }

  double dual(const Eigen::VectorXd& theta, const Eigen::VectorXd& y) const
  {
    return 0.5 * y.squaredNorm() - 0.5 * (theta + y).squaredNorm();
  }

  Eigen::VectorXd residual(const Eigen::VectorXd& eta,
                           const Eigen::VectorXd& y) const
  {
    return eta - y;
  }

  void updateResidual(Eigen::VectorXd& residual,
                      const Eigen::VectorXd& eta,
                      const Eigen::VectorXd& y) const
  {
    residual = eta - y;
  }

  template<typename T>
  void updateResidual(Eigen::VectorXd& residual,
                      const double beta_diff,
                      const T& x,
                      const size_t j) const
  {
    residual += x.col(j) * beta_diff;
  }

  void updateResidual(Eigen::VectorXd& residual,
                      const double intercept_update) const
  {
    residual.array() -= intercept_update;
  }

  template<typename T>
  Eigen::MatrixXd hessian(const T& x,
                          const std::vector<size_t>& ind_a,
                          const std::vector<size_t>& ind_b) const
  {
    // TODO(jolars): will be more efficient with slicing from Eigen 3.4
    Eigen::MatrixXd out(ind_a.size(), ind_b.size());
    for (size_t i = 0; i < ind_a.size(); ++i) {
      for (size_t j = 0; j < ind_b.size(); ++j) {
        out(i, j) = x.col(ind_a[i]).dot(x.col(ind_b[j]));
        // out(j, i) = out(i, j);
      }
    }

    return out;
  }

  template<typename T>
  double hessianTerm(const T& x, const size_t j)
  {
    if (hessian_cache[j] == -1)
      hessian_cache[j] = x.col(j).squaredNorm();

    return hessian_cache[j];
  }
};

} // namespace hesso
