#pragma once

#include "gaussian.h"
#include "objective.h"
#include "prox.h"
#include "results.h"
#include "set_operations.h"
#include "utils.h"
// #include <Eigen/Core>
// #include <Eigen/Eigenvalues>
#include <RcppEigen.h>
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
       Eigen::VectorXd& theta,
       Eigen::VectorXd& residual,
       Eigen::VectorXd& gradient,
       const Eigen::VectorXd& y,
       const double null_primal,
       const double lambda,
       const double tol,
       const size_t max_it)
{
  const size_t p = x.cols();

  std::vector<double> primals;
  std::vector<double> duals;
  std::vector<double> gaps;

  double gap{ 0 };

  size_t it{ 0 };

  for (; it < max_it + 1; ++it) {
    gradient = x.transpose() * residual;
    double primal = objective.loss(residual) + lambda * beta.lpNorm<1>();

    // retrieve feasible dual point, theta, by dual scaling
    theta = residual;
    theta /= std::max(1.0, gradient.lpNorm<Eigen::Infinity>() / lambda);
    double dual = objective.dual(theta, y);
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
      break;
    }

    for (size_t j = 0; j < p; ++j) {
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
  VectorXd theta{ VectorXd::Zero(n) };
  VectorXd residual{ -y };
  VectorXd gradient{ x.transpose() * residual };

  std::vector<size_t> active;
  std::vector<size_t> ever_active;

  std::vector<size_t> passes;
  std::vector<double> gaps;

  std::vector<bool> screened;
  std::vector<bool> strong;

  double lambda_max{ 0 };
  size_t first_active;

  for (long int i = 0; i < gradient.size(); ++i) {
    if (std::abs(gradient(i)) >= lambda_max) {
      first_active = i;
      lambda_max = std::abs(gradient(i));
    }
  }

  active.emplace_back(first_active);

  MatrixXd h = objective.hessian(x, active, active);
  MatrixXd h_inv = h.inverse();
  VectorXi s = VectorXi::Zero(p);
  s(first_active) = -signum(gradient(active[0]));

  const double null_primal =
    objective.loss(residual) + lambda_max * beta.lpNorm<1>();

  for (long int step = 0; step < lambda.size(); ++step) {
    // print(active, "active");

    VectorXd h_inv_s = VectorXd::Zero(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
      h_inv_s += h_inv.col(i) * s(active[i]);
    }

    // warm start
    if (warm_starts) {
      if (step > 0) {
        for (size_t i = 0; i < active.size(); ++i) {
          beta(active[i]) += (lambda(step - 1) - lambda(step)) * h_inv_s(i);
        }
        objective.updateResidual(residual, x * beta, y);
      }
    }

    auto solver_results = solver(x,
                                 objective,
                                 beta,
                                 theta,
                                 residual,
                                 gradient,
                                 y,
                                 null_primal,
                                 lambda(step),
                                 tol,
                                 max_it);

    betas.col(step) = beta;
    gaps.emplace_back(solver_results.gap);
    passes.emplace_back(solver_results.passes);

    // to keep track of Hessian updates
    if (step > 0) {
      std::vector<size_t> drop, keep;

      Rcpp::Rcout << "beta " << beta << "\n";

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

      std::vector<size_t> active_new;

      for (long int j = 0; j < beta.size(); ++j) {
        if (beta(j) != 0) {
          s(j) = -signum(gradient(j));
          // s(j) = signum(beta(j));

          if (!contains(active, j)) {
            active_new.emplace_back(j);
          }

          if (!contains(ever_active, j)) {
            ever_active.emplace_back(j);
          }
        } else {
          s(j) = 0;
        }
      }

      // print(active_new, "active_new");
      // print(keep, "keep");
      // print(drop, "drop");

      std::sort(ever_active.begin(), ever_active.end());

      bool verify_hessian = true;
      bool verbose = true;

      updateHessian(h,
                    h_inv,
                    x,
                    objective,
                    active,
                    active_new,
                    keep,
                    drop,
                    verify_hessian,
                    verbose);

      // Rcpp::Rcout << h << std::endl;
      // Rcpp::Rcout << h_inv << std::endl;

      active.insert(active.end(), active_new.cbegin(), active_new.cend());
    }
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
