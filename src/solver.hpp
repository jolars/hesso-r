#pragma once

#include "kktCheck.h"
#include "linearAlgebra.h"
#include "model.h"
#include "safeScreening.h"
#include "updateCorrelation.h"
#include "updateLinearPredictor.h"
#include "utils.h"
#include <RcppArmadillo.h>
#include <memory>

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

  bool inner_solver_converged = false;

  vec c_old(p);
  vec d(p);
  vec residual_old;

  double tol_mod = model->toleranceModifier(y);

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
        inner_solver_converged = false;

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

        if (duality_gap <= tol_gap * tol_mod) {
          break;
        }

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

      Rcpp::checkUserInterrupt();

      if (it_inner % check_frequency == 0) {
        primal_value =
          model->primal(residual, Xbeta, beta, y, lambda, working_set);

        updateCorrelation(c, residual, X, working_set, X_offset, standardize);

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

        inner_solver_converged =
          duality_gap <= tol_gap_inner || primal_value >= primal_value_prev;

        if (inner_solver_converged) {
          if (verbosity >= 2)
            Rprintf("      inner solver converged\n");

          primal_value_prev = datum::inf;
        } else {
          primal_value_prev = primal_value;
        }
      }

      double t0 = timer.toc();

      if (!inner_solver_converged) {
        if (it_inner != 0) {
          // if (shuffle || !progress)
          if (shuffle)
            working_set = arma::shuffle(working_set);
        }

        for (auto&& j : working_set) {
          updateCorrelation(c, residual, X, j, X_offset, standardize);
          double hess_j = model->hessianTerm(X, j, X_offset, standardize);

          if (hess_j <= 0)
            continue;

          double beta_j_old = beta(j);
          double beta_j_new = prox(beta_j_old + c(j) / hess_j, lambda / hess_j);
          double beta_j_diff = beta_j_new - beta_j_old;

          if (beta_j_diff != 0) {
            beta(j) = beta_j_new;
            model->adjustResidual(
              residual, Xbeta, X, y, j, beta_j_diff, X_offset, standardize);
          }
        }

        cd_time += timer.toc() - t0;

      }

      if (it % 10 == 0) {
        Rcpp::checkUserInterrupt();
      }

      it_inner++;
      it++;
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
