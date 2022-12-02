#' Fit a Lasso Path
#'
#' @param x Design matrix. If sparse needs to be of class `dgCMatrix`. If dense,
#'   needs to be a matrix (i.e. not a `data.frame`).
#' @param y Response vector
#' @param intercept Whether to fit an intercept
#' @param lambda Lambda sequence
#' @param path_length The length of the path. Ignored if `lambda` is not `NULL`.
#' @param lambda_min_ratio A fraction of the largest lambda in the path, which
#'   is the last point on the path. Ignored if `lambda` is not NULL.
#' @param tol Stopping tolerance in terms of the duality gap relative to the
#'   value of the primal objective for the null model
#' @param max_it Number of maximum iterations for the solver
#'
#' @return family Type of objective
#' @export
lasso <- function(x,
                  y,
                  intercept = TRUE,
                  lambda = NULL,
                  path_length = 100,
                  lambda_min_ratio = if (NROW(x) > NCOL(x)) 1e-4 else 1e-2,
                  tol = 1e-6,
                  max_it = 10000,
                  warm_starts = TRUE) {
  stopifnot(
    is.logical(warm_starts),
    is.logical(intercept),
    is.numeric(path_length),
    is.numeric(tol),
    is.numeric(path_length),
    is.numeric(max_it),
    path_length >= 1,
    lambda_min_ratio > 0,
    lambda_min_ratio < 1,
    tol > 0,
    max_it >= 0
  )

  if (is.null(lambda)) {
    lambda <- double(0)
  } else {
    stopifnot(
      is.numeric(lambda),
      is.finite(lambda),
      lambda > 0
    )
  }

  args <- list(
    intercept = intercept,
    path_length = path_length,
    lambda_min_ratio = lambda_min_ratio,
    tol = tol,
    max_it = max_it,
    warm_starts = warm_starts
  )

  if (inherits(x, "sparseMatrix")) {
    if (!inherits(x, "dgCMatrix")) {
      stop("Sparse matrices need to be of class 'dgcMatrix'")
    }
    rcppLassoSparse(x, y, lambda, args)
  } else {
    if (!inherits(x, "matrix")) {
      stop("'x' needs to be a matrix")
    }
    rcppLassoDense(x, y, lambda, args)
  }
}
