#pragma once

#include "kktCheck.h"
#include "linearAlgebra.h"
#include "model.h"
#include "safeScreening.h"
#include "updateCorrelation.h"
#include "updateLinearPredictor.h"
#include "utils.h"
#include <RcppArmadillo.h>

template<typename T>
std::tuple<double,
           double,
           double,
           arma::vec,
           arma::uword,
           double,
           arma::uword,
           arma::uword,
           double,
           double>
fit(arma::uvec& screened,
    arma::vec& c,
    arma::vec& residual,
    arma::vec& Xbeta,
    arma::vec& beta,
    const std::unique_ptr<Model>& model,
    const T& X,
    const arma::vec& y,
    const arma::vec& X_norms,
    const arma::vec& X_offset,
    const bool standardize,
    const arma::uvec& active_set,
    const arma::uvec& strong_set,
    const double lambda,
    const double lambda_prev,
    const double lambda_max,
    const arma::uword n_active_prev,
    const bool shuffle,
    const arma::uword check_frequency,
    const bool augment_with_gap_safe,
    const arma::uword step,
    const arma::uword maxit,
    const double tol_gap,
    const bool line_search,
    const arma::uword verbosity)
{
  using namespace arma;

  const uword n = X.n_rows;
  const uword p = X.n_cols;

  uvec screened_set = find(screened);
  uvec working_set = screened_set;
  uvec safe(p, fill::ones);
  uvec safe_set = find(safe);
  uvec violations(p);

  uword n_refits{ 0 };
  uword n_violations{ 0 };

  double primal_value =
    model->primal(residual, Xbeta, beta, y, lambda, screened_set);
  double primal_value_prev = primal_value;

  double dual_scale = std::max(lambda, max(abs(c)));
  vec theta = residual / dual_scale;
  double dual_value = model->dual(theta, y, lambda);

  vec w(n, fill::ones);

  double duality_gap = primal_value - dual_value;
  double duality_gap_prev = datum::inf;

  bool inner_solver_converged = false;
  bool progress = true;

  vec c_old(p);
  vec d(p);
  vec residual_old;

  double tol_mod = model->toleranceModifier(y);

  // line search parameters
  const uword MAX_BACKTRACK_ITR = 20;
  const uword MAX_PROX_NEWTON_CD_ITR = 20;
  const double PROX_NEWTON_EPSILON_RATIO = 10;
  const uword MIN_PROX_NEWTON_CD_ITR = 2;
  double prox_newton_grad_diff = 0;

  double n_screened = 0;
  uword it = 0;
  uword it_inner = 0;
  double tol_gap_inner = tol_gap * tol_mod;

  // timing
  wall_clock timer;
  timer.tic();

  double cd_time = 0;
  double kkt_time = 0;

  if (!screened_set.is_empty()) {
    updateLinearPredictor(Xbeta, X, beta, X_offset, standardize, active_set);
    model->updateResidual(residual, Xbeta, y);

    while (it < maxit) {
      if (verbosity >= 2) {
        Rprintf("    iter: %i\n", it + 1);
      }

      if (inner_solver_converged && it > 0) {
        double t0 = timer.toc();

        bool outer_check = false;

        uvec unscreened_set = find(safe && (screened == false));

        violations.fill(false);

        uvec check_set;

        check_set = setIntersect(unscreened_set, strong_set);
        updateCorrelation(c, residual, X, check_set, X_offset, standardize);
        kktCheck(violations, screened, c, check_set, lambda);

        if (!any(violations)) {
          check_set = check_set = setDiff(unscreened_set, strong_set);
          updateCorrelation(c, residual, X, check_set, X_offset, standardize);
          kktCheck(violations, screened, c, check_set, lambda);

          outer_check = true;
        }

        screened_set = find(screened);

        kkt_time += timer.toc() - t0;

        // check duality gap even if there are violations
        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, working_set);

        dual_scale = std::max(lambda, max(abs(c(safe_set))));
        theta = residual / dual_scale;
        dual_value = model->dual(theta, y, lambda);
        duality_gap = primal_value - dual_value;

        if (verbosity >= 2) {
          Rprintf("    GLOBAL primal: %f, dual: %f, gap: %f, tol: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap,
                  tol_gap * tol_mod);
        }

        if (duality_gap <= tol_gap * tol_mod)
          break;

        if (any(violations)) {
          n_refits += 1;
        }

        if (outer_check) {
          // have updated entire correlation vector, so let's use
          // it to gap-safe screen too (at no cost)
          double r_screen = model->safeScreeningRadius(duality_gap, lambda);

          safeScreening(safe,
                        safe_set,
                        residual,
                        Xbeta,
                        beta,
                        c,
                        dual_scale,
                        r_screen,
                        model,
                        X,
                        y,
                        X_offset,
                        standardize,
                        X_norms);

          screened = screened && safe;
          screened_set = find(screened);
        }

        working_set = screened_set;

        n_violations += sum(violations);
      }

      n_screened += screened_set.n_elem;

      if (inner_solver_converged) {
        // reset inner solver variables
        it_inner = 0;
        inner_solver_converged = false;
        progress = true;
      }

      double t0 = timer.toc();

      if (!line_search) {
        if (it_inner != 0) {
          if (shuffle || !progress)
            working_set = arma::shuffle(working_set);
        }

        for (auto&& j : working_set) {
          updateCorrelation(c, residual, X, j, X_offset, standardize);
          double hess_j = model->hessianTerm(X, j, X_offset, standardize);

          if (hess_j <= 0)
            continue;

          double beta_j_old = beta(j);
          double v =
            prox(beta_j_old + c(j) / hess_j, lambda / hess_j) - beta(j);

          if (v != 0) {
            beta(j) = beta_j_old + v;
            model->adjustResidual(residual,
                                  Xbeta,
                                  X,
                                  y,
                                  j,
                                  beta(j) - beta_j_old,
                                  X_offset,
                                  standardize);
          }
        }
      } else {
        // blitz-type line search
        // this code is based on https://github.com/tbjohns/BlitzL1 as of
        // 2022-01-12, which is licensed under the MIT license, Copyright
        // Tyler B. Johnson 2015
        uword ws_size = working_set.n_elem;

        vec X_delta_beta(n, fill::zeros);
        vec delta_beta(ws_size, fill::zeros);
        vec hess_cache(ws_size);
        vec prox_newton_grad_cache(ws_size);

        double prox_newton_epsilon = 0;

        uword max_cd_itr = MAX_PROX_NEWTON_CD_ITR;

        w = model->weights(residual, y);

        for (uword j = 0; j < ws_size; ++j) {
          uword ind = working_set(j);
          hess_cache(j) = model->hessianTerm(X, ind, X_offset, standardize);
        }

        if (it_inner == 0) {
          max_cd_itr = MIN_PROX_NEWTON_CD_ITR;
          prox_newton_grad_diff = 0;

          updateCorrelation(c, residual, X, working_set, X_offset, standardize);
          prox_newton_grad_cache = -c(working_set);
        } else {
          prox_newton_epsilon =
            PROX_NEWTON_EPSILON_RATIO * prox_newton_grad_diff;
        }

        for (uword it_inner = 0; it_inner < max_cd_itr; ++it_inner) {
          // shuffle the working set
          uvec perm = randperm(ws_size);
          working_set = working_set(perm);
          delta_beta = delta_beta(perm);
          prox_newton_grad_cache = prox_newton_grad_cache(perm);
          hess_cache = hess_cache(perm);

          double sum_sq_hess_diff = 0;

          for (uword j = 0; j < ws_size; ++j) {
            uword ind = working_set(j);
            double hess_j = hess_cache(j);

            if (hess_j <= 0)
              continue;

            double grad = prox_newton_grad_cache(j) +
                          weightedInnerProduct(
                            X, ind, X_delta_beta, w, X_offset, standardize);

            double old_value = beta(ind) + delta_beta(j);
            double proposal = old_value - grad / hess_j;
            double new_value = prox(proposal, lambda / hess_j);
            double diff = new_value - old_value;

            if (diff != 0) {
              delta_beta(j) = new_value - beta(ind);

              addScaledColumn(
                X_delta_beta, X, ind, diff, X_offset, standardize);
              sum_sq_hess_diff += diff * diff * hess_j * hess_j;
            }
          }

          if (sum_sq_hess_diff < prox_newton_epsilon &&
              it_inner + 1 >= MIN_PROX_NEWTON_CD_ITR) {
            break;
          }
        }

        double t = 1;
        double last_t = 0;

        for (uword backtrack_itr = 0; backtrack_itr < MAX_BACKTRACK_ITR;
             ++backtrack_itr) {
          double diff_t = t - last_t;

          double subgrad_t = 0;

          for (uword j = 0; j < ws_size; ++j) {
            uword ind = working_set(j);
            beta(ind) += diff_t * delta_beta(j);

            if (beta(ind) < 0)
              subgrad_t -= lambda * delta_beta(j);
            else if (beta(ind) > 0)
              subgrad_t += lambda * delta_beta(j);
            else
              subgrad_t -= lambda * std::abs(delta_beta(j));
          }

          Xbeta += diff_t * X_delta_beta;

          model->updateResidual(residual, Xbeta, y);

          subgrad_t += dot(X_delta_beta, -residual);

          if (subgrad_t < 0) {
            break;
          } else {
            last_t = t;
            t *= 0.5;
          }
        }

        // cache gradients for next iteration
        if (t != 1) {
          X_delta_beta *= t;
        }

        updateCorrelation(c, residual, X, working_set, X_offset, standardize);

        for (uword j = 0; j < ws_size; ++j) {
          uword ind = working_set(j);

          double actual_grad = -c(ind);
          double approximate_grad =
            prox_newton_grad_cache(j) +
            weightedInnerProduct(
              X, ind, X_delta_beta, w, X_offset, standardize);

          prox_newton_grad_cache(j) = actual_grad;
          double diff = actual_grad - approximate_grad;
          prox_newton_grad_diff += diff * diff;
        }
      }

      if ((it_inner % check_frequency == 0) || line_search) {
        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, working_set);

        if (!line_search) {
          // correlation vector is always updated at end of line search
          updateCorrelation(c, residual, X, working_set, X_offset, standardize);
        }

        dual_scale = std::max(lambda, max(abs(c(working_set))));
        theta = residual / dual_scale;
        dual_value = model->dual(theta, y, lambda);

        duality_gap = primal_value - dual_value;

        if (verbosity >= 2)
          Rprintf("      primal: %f, dual: %f, gap: %f, tol: %f\n",
                  primal_value,
                  dual_value,
                  duality_gap,
                  tol_gap_inner);

        inner_solver_converged = duality_gap <= tol_gap_inner;

        if (line_search) {
          // line search should ensure progress in the primal, so if primal
          // value increases, then the limit of machine precision must have been
          // reached
          if (primal_value >= primal_value_prev)
            inner_solver_converged = true;
        }

        progress = duality_gap < duality_gap_prev;

        if (!progress && verbosity >= 2)
          Rprintf("      no progress; shuffling indices\n");

        if (inner_solver_converged) {
          if (verbosity >= 2)
            Rprintf("      inner solver converged\n");

          primal_value_prev = datum::inf;
          duality_gap_prev = datum::inf;
        } else {
          primal_value_prev = primal_value;
          duality_gap_prev = duality_gap;
        }
      }

      cd_time += timer.toc() - t0;

      if (it % 10 == 0) {
        Rcpp::checkUserInterrupt();
      }

      it++;
      it_inner++;
    }

  } else {
    beta.zeros();
  }

  // Return average number of screened predictors. If it == 0, then the
  // algorithm converged instantly and hence did not need to do any screening
  double avg_screened = it == 0 ? 0 : n_screened / it;

  return { primal_value, dual_value,   duality_gap, theta,   it,
           avg_screened, n_violations, n_refits,    cd_time, kkt_time };
}
