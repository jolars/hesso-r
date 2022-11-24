#' Lasso Path with Hessian Screening Rules
#'
#' This function fits the full lasso path.
#'
#' @param X The predictor matrix
#' @param y The reponse vector
#' @param family The name of the family, "gaussian" or "logistic"
#' @param standardize Whether to standardize the predictors
#' @param shuffle Shuffle working set before each CD pass?
#' @param check_frequency Frequency at which duality gap is checked for
#'   inner CD loop
#' @param hessian_warm_starts Whether to use warm starts based on Hessian
#' @param augment_with_gap_safe Whether or not augment heuristic rules
#'   with Gap Safe checks during KKT checks
#' @param log_hessian_update_type What type of strategy to use for
#'   updating the hessian for logistic regression
#' @param path_length The (desired) length of the lasso path
#' @param maxit Maximum number of iterations for Coordinate Descent loop
#' @param tol_gap Tolerance threshold for relative duality gap.
#' @param gamma Percent of strong approximation to add to Hessian approximation
#' @param verify_hessian Whether to not to verify that Hessian updates are
#'   correct. Used only for diagnostic purposes.
#' @param line_search Use line search in CD solver.
#' @param verbosity Controls the level of verbosity. 0 = no output, 1 = outer
#'   level output, 2 = inner solver output
#' @param lambda weights for the regularization path, if `NULL`, then they
#'   are automatically computed
#' @param store_dual_variables whether to store dual variables throughout
#'   fitting
#'
#' @export
lasso_path <- function(x,
                       y,
                       family = c("gaussian", "binomial"),
                       lambda = NULL,
                       standardize = TRUE,
                       shuffle = FALSE,
                       check_frequency = if (NROW(x) > NCOL(x)) 1 else 10,
                       hessian_warm_starts = TRUE,
                       augment_with_gap_safe = TRUE,
                       log_hessian_update_type = c("full", "approx"),
                       log_hessian_auto_update_freq = 10,
                       path_length = 100L,
                       maxit = 1e5,
                       tol_gap = 1e-4,
                       gamma = 0.01,
                       store_dual_variables = FALSE,
                       verify_hessian = FALSE,
                       line_search = TRUE,
                       verbosity = 0) {
  family <- match.arg(family)
  log_hessian_update_type <- match.arg(log_hessian_update_type)

  if (is.null(lambda)) {
    lambda <- double(path_length)
    lambda_type <- "auto"
  } else {
    lambda_type <- "user"
  }

  sparse <- inherits(x, "sparseMatrix")

  if (sparse) {
    x <- as(x, "dgCMatrix")

    lassoPathSparse(
      x,
      y,
      family,
      lambda,
      lambda_type,
      standardize,
      shuffle,
      check_frequency,
      hessian_warm_starts,
      augment_with_gap_safe,
      log_hessian_update_type,
      path_length,
      maxit,
      tol_gap,
      gamma,
      store_dual_variables,
      verify_hessian,
      line_search,
      verbosity
    )
  } else {
    lassoPathDense(
      x,
      y,
      family,
      lambda,
      lambda_type,
      standardize,
      shuffle,
      check_frequency,
      hessian_warm_starts,
      augment_with_gap_safe,
      log_hessian_update_type,
      path_length,
      maxit,
      tol_gap,
      gamma,
      store_dual_variables,
      verify_hessian,
      line_search,
      verbosity
    )
  }
}
