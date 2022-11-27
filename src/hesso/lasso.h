#pragma once

#include "gaussian.h"
#include "kkt_check.h"
#include "objective.h"
#include "prox.h"
#include "results.h"
#include "set_operations.h"
#include "utils.h"
// #include <Eigen/Core>
// #include <Eigen/Eigenvalues>
#include <RcppEigen.h>
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <memory>
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

template<typename T, typename P>
Results
lasso(const T& x,
      P& objective,
      const Eigen::VectorXd& y,
      const Eigen::ArrayXd& lambda,
      const double tol,
      const size_t max_it,
      const bool warm_starts)
{
  using namespace Eigen;

  const size_t n = x.rows();
  const size_t p = x.cols();

  MatrixXd betas(p, lambda.size());

  VectorXd beta{ VectorXd::Zero(p) };
  VectorXd residual{ -y };
  VectorXd eta{ VectorXd::Zero(n) };
  VectorXd gradient{ x.transpose() * residual };

  std::vector<size_t> active;

  std::vector<size_t> passes;
  std::vector<double> gaps;

  boost::dynamic_bitset<> ever_active(p);
  boost::dynamic_bitset<> working(p);
  boost::dynamic_bitset<> strong(p);

  double lambda_max{ 0 };
  size_t first_active;

  for (long int i = 0; i < gradient.size(); ++i) {
    if (std::abs(gradient(i)) >= lambda_max) {
      first_active = i;
      lambda_max = std::abs(gradient(i));
    }
  }

  active.emplace_back(first_active);
  ever_active[first_active] = true;

  MatrixXd h = objective.hessian(x, active, active);
  MatrixXd h_inv = h.inverse();

  VectorXi s = VectorXi::Zero(p);
  s(first_active) = -signum(gradient(first_active));

  const double null_primal =
    objective.loss(residual) + lambda_max * beta.lpNorm<1>();

  double lambda_prev = lambda_max;

  for (Index step = 0; step < lambda.size(); ++step) {
    double lambda_next = lambda(step);

    working = ever_active;

    // screening
    strong.reset();
    for (size_t j = 0; j < p; ++j) {
      if (gradient(j) >= 2 * lambda_next - lambda_prev) {
        strong[j] = true;
      }
    }

    // warm start
    VectorXd h_inv_s = VectorXd::Zero(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
      h_inv_s += h_inv.col(i) * s(active[i]);
    }

    if (warm_starts) {
      for (size_t j = 0; j < active.size(); ++j) {
        beta(active[j]) += (lambda_prev - lambda_next) * h_inv_s(j);
      }

      eta.setZero();
      for (const auto& j : active) {
        eta += x.col(j) * beta(j);
      }
      objective.updateResidual(residual, x * beta, y);
    }

    auto solver_results = solver(x,
                                 objective,
                                 beta,
                                 residual,
                                 gradient,
                                 working,
                                 strong,
                                 y,
                                 null_primal,
                                 lambda(step),
                                 tol,
                                 max_it);

    betas.col(step) = beta;
    gaps.emplace_back(solver_results.gap);
    passes.emplace_back(solver_results.passes);

    // to keep track of Hessian updates
    std::vector<size_t> drop, keep;

    // first find out which of the current features are kept or lost
    for (size_t j = 0; j < active.size(); ++j) {
      if (beta(active[j]) != 0) {
        keep.emplace_back(j);
      } else {
        drop.emplace_back(j);
        active[j] = -1; // mark for removal
      }
    }

    active.erase(std::remove(active.begin(), active.end(), -1), active.end());

    print(active, "active");

    std::vector<size_t> active_new;

    for (long int j = 0; j < beta.size(); ++j) {
      if (beta(j) != 0) {
        s(j) = -signum(gradient(j));

        if (!contains(active, j))
          active_new.emplace_back(j);

        ever_active[j] = true;
      } else {
        s(j) = 0;
      }
    }

    updateHessian(
      h, h_inv, x, objective, active, active_new, keep, drop, true, true);

    active.insert(active.end(), active_new.cbegin(), active_new.cend());
  }

  return Results{ betas, gaps, passes };
}

template<typename T>
Results
lassoWrapper(const T& x,
             const Eigen::VectorXd& y,
             const Eigen::ArrayXd& lambda,
             const double tol,
             const size_t max_it,
             const bool warm_starts)
{
  size_t p = x.cols();
  Objective<Gaussian> objective{ p };
  return lasso(x, objective, y, lambda, tol, max_it, warm_starts);
}
}
