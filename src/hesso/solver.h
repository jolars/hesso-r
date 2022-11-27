#pragma once

#include "kkt_check.h"
#include "objective.h"
#include "prox.h"
#include <RcppEigen.h>
#include <boost/dynamic_bitset.hpp>
#include <vector>

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
       Eigen::VectorXd& residual,
       Eigen::VectorXd& gradient,
       boost::dynamic_bitset<>& working,
       boost::dynamic_bitset<>& strong,
       const Eigen::VectorXd& y,
       const double null_primal,
       const double lambda,
       const double tol,
       const size_t max_it)
{

  std::vector<double> primals;
  std::vector<double> duals;
  std::vector<double> gaps;

  double gap{ 0 };

  size_t it{ 0 };

  for (; it < max_it + 1; ++it) {
    double max_abs_grad = 0.0;
    for (size_t j = working.find_first(); j != boost::dynamic_bitset<>::npos;
         j = working.find_next(j)) {
      gradient(j) = x.col(j).dot(residual);
      max_abs_grad = std::max(std::abs(gradient(j)), max_abs_grad);
    }
    double primal = objective.loss(residual) + lambda * beta.lpNorm<1>();

    // retrieve feasible dual point by dual scaling
    double dual_scaling = std::max(1.0, max_abs_grad / lambda);
    double dual = objective.dual(residual / dual_scaling, y);
    gap = primal - dual;

    // Rprintf("it: %i primal: %f, dual: %f, gap: %f, tol: %f\n",
    //         it,
    //         primal,
    //         dual,
    //         gap,
    //         tol);

    primals.emplace_back(primal);
    duals.emplace_back(dual);
    gaps.emplace_back(gap);

    // if (gap <= tol * null_primal || it == max_it) {
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

      // Rprintf("  j: %i, gradient: %f, hess: %f, beta_old: %f, beta_new:
      // %f\n",
      //         j,
      //         gradient(j),
      //         1 / t,
      //         beta_j_old,
      //         beta_j_new);

      if (beta_j_new != beta_j_old) {
        beta(j) = beta_j_new;
        objective.updateResidual(residual, beta_j_new - beta_j_old, x, j);
      }
    }
  }

  return { gap, it };
}
}
