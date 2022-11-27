#' Fit a Lasso Path
#'
#' @param x Design matrix
#' @param y Response vector
#' @param lambda Lambda sequence
#' @param tol Stopping tolerance in terms of the duality gap relative to the
#'   value of the primal objective for the null model
#' @param max_it Number of maximum iterations for the solver
#'
#' @return family Type of objective
#' @export
lasso <- function(x,
                  y,
                  lambda = NULL,
                  path_length = 100,
                  lambda_min_ratio = if (NROW(x) > NCOL(x)) 1e-4 else 1e-2,
                  tol = 1e-6,
                  max_it = 10000,
                  warm_starts = TRUE) {
  if (is.null(lambda)) {
    lambda <- double(0)
  }

  args <- list(
    path_length = path_length,
    lambda_min_ratio = lambda_min_ratio,
    tol = tol,
    max_it = max_it,
    warm_starts = warm_starts
  )

  rcppLassoDense(x, y, lambda, args)
}
