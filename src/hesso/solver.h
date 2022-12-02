#pragma once

#include "kkt_check.h"
#include "objective.h"
#include "prox.h"
#include <RcppEigen.h>
#include <boost/dynamic_bitset.hpp>

namespace hesso {

struct SolverResults
{
  const double gap;
  const size_t passes;
};

template<typename T, typename P>
SolverResults
solver(const T& x,
       P& objective,
       Eigen::VectorXd& beta,
       double& beta0,
       Eigen::VectorXd& residual,
       Eigen::VectorXd& gradient,
       boost::dynamic_bitset<>& working,
       boost::dynamic_bitset<>& strong,
       const Eigen::VectorXd& y,
       const bool intercept,
       const double null_primal,
       const double lambda,
       const double tol,
       const size_t max_it)
{

  double gap{ 0 };

  size_t it{ 0 };

  for (; it < max_it + 1; ++it) {
    double max_abs_grad = 0.0;
    for (size_t j = working.find_first(); j != boost::dynamic_bitset<>::npos;
         j = working.find_next(j)) {
      gradient(j) = x.col(j).dot(residual);
      max_abs_grad = std::max(std::abs(gradient(j)), max_abs_grad);
    }
    Eigen::VectorXd eta = (x * beta).array() + beta0;
    objective.updateResidual(residual, eta, y);
    double primal = objective.loss(residual) + lambda * beta.lpNorm<1>();

    // retrieve feasible dual point by dual scaling
    gradient = x.transpose() * residual;
    double dual_scaling = std::max(1.0, max_abs_grad / lambda);
    double dual = objective.dual(residual / dual_scaling, y);
    gap = primal - dual;

    if (gap <= tol || it == max_it) {
      bool any_violations = checkKktConditions(
        working, gradient, ~working & strong, x, residual, lambda);
      if (!any_violations) {
        any_violations = checkKktConditions(
          working, gradient, ~(working | strong), x, residual, lambda);
        if (!any_violations) {
          break;
        }
      }
    }

    for (size_t j = working.find_first(); j != boost::dynamic_bitset<>::npos;
         j = working.find_next(j)) {
      double grad_j = x.col(j).dot(residual);
      double t = 1.0 / objective.hessianTerm(x, j);
      double beta_j_old = beta(j);
      double beta_j_new = prox(beta_j_old - t * grad_j, t * lambda);

      if (beta_j_new != beta_j_old) {
        beta(j) = beta_j_new;
        objective.updateResidual(residual, beta_j_new - beta_j_old, x, j);
      }
    }

    if (intercept) {
      double intercept_update = residual.mean();
      objective.updateResidual(residual, intercept_update);
      beta0 -= intercept_update;
    }
  }

  return { gap, it };
}
}
