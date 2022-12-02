#pragma once

#include "math.h"
#include "objective.h"
#include "results.h"
#include "solver.h"
#include "utils.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <vector>

namespace hesso {

template<typename T, typename P>
Results
lasso(const T& x,
      P& objective,
      const Eigen::VectorXd& y,
      const bool intercept,
      std::vector<double> lambda,
      const size_t path_length,
      const double lambda_min_ratio,
      const double tol,
      const size_t max_it,
      const bool warm_starts)
{
  using namespace Eigen;

  const size_t n = x.rows();
  const size_t p = x.cols();

  double beta0 = intercept ? y.mean() : 0.0;

  VectorXd beta{ VectorXd::Zero(p) };
  VectorXd eta{ n };
  eta.setConstant(beta0);
  VectorXd residual = objective.residual(eta, y);
  VectorXd gradient{ x.transpose() * residual };

  std::vector<size_t> active;

  std::vector<size_t> passes;
  std::vector<double> gaps;

  boost::dynamic_bitset<> ever_active(p);
  boost::dynamic_bitset<> working(p);
  boost::dynamic_bitset<> strong(p);

  double lambda_max{ 0 };
  size_t first_active;

  for (Index i = 0; i < gradient.size(); ++i) {
    if (std::abs(gradient(i)) >= lambda_max) {
      first_active = i;
      lambda_max = std::abs(gradient(i));
    }
  }

  lambda_max = gradient.lpNorm<Eigen::Infinity>();

  bool automatic_lambda = lambda.empty();

  if (automatic_lambda) {
    // automatically generate lambda
    lambda = geomSpace(lambda_max, lambda_min_ratio * lambda_max, path_length);
  }

  std::vector<double> beta0s;
  MatrixXd betas(p, lambda.size());

  active.emplace_back(first_active);
  ever_active[first_active] = true;

  MatrixXd h = objective.hessian(x, active, active);
  MatrixXd h_inv = h.inverse();

  VectorXi s = VectorXi::Zero(p);
  s(first_active) = -signum(gradient(first_active));

  const double null_primal =
    objective.loss(residual) + lambda_max * beta.lpNorm<1>();

  const double dev_null = objective.deviance(residual);
  double dev_prev = dev_null * 2;

  double lambda_prev = lambda_max;

  Index step = 0;

  for (; step < static_cast<int>(lambda.size()); ++step) {
    double lambda_current = lambda[step];

    working = ever_active;

    // screening
    strong.reset();
    for (size_t j = 0; j < p; ++j) {
      if (gradient(j) >= 2 * lambda_current - lambda_prev) {
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
        beta(active[j]) += (lambda_prev - lambda_current) * h_inv_s(j);
      }

      eta.setConstant(beta0);
      for (const auto& j : active) {
        eta += x.col(j) * beta(j);
      }
      objective.updateResidual(residual, eta, y);

      if (intercept) {
        double intercept_update = residual.mean();
        objective.updateResidual(residual, intercept_update);
        beta0 -= intercept_update;
      }
    }

    auto solver_results = solver(x,
                                 objective,
                                 beta,
                                 beta0,
                                 residual,
                                 gradient,
                                 working,
                                 strong,
                                 y,
                                 intercept,
                                 null_primal,
                                 lambda[step],
                                 tol,
                                 max_it);

    betas.col(step) = beta;
    beta0s.emplace_back(beta0);
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

    if (automatic_lambda) {
      // check stopping conditions
      double dev = objective.deviance(residual);
      double dev_ratio = 1 - dev / dev_null;
      double dev_change = 1 - dev / dev_prev;

      if (dev_ratio >= 0.999 || dev_change < 1e-5 || active.size() > n) {
        break;
      }
    } else if (step == static_cast<Index>(lambda.size()) - 1) {
      break;
    }

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

    updateHessian(h, h_inv, x, objective, active, active_new, keep, drop);

    active.insert(active.end(), active_new.cbegin(), active_new.cend());
  }

  lambda.erase(std::cbegin(lambda) + step, std::cend(lambda));

  return Results{ betas, beta0s, gaps, lambda, passes };
}
}
