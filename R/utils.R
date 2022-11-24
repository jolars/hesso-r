#' Get Duality Gaps
#'
#' @param fit the resulting fit
#' @param x design matrix
#' @param y response vector
#' @param standardize whether the fit used standardization
#' @param tol_gap the tolerance threshold used when fitting
#'
#' @return primals, duals, and relative duality gaps
#' @export
check_gaps <- function(beta,
                       lambda,
                       x,
                       y,
                       standardize,
                       tol_gap = 1e-4) {
  if (standardize) {
    pop_sd <- function(a) sqrt((length(a) - 1) / length(a)) * sd(a)

    s <- apply(x, 2, pop_sd)
    beta <- beta * s

    x <- scale(x, scale = s)
  }

  duals <- primals <- double(length(lambda))

  n <- length(y)

  for (i in seq_along(duals)) {
    if (family == "gaussian") {
      residual <- y - x %*% beta[, i]
      dual_scale <-
        max(1, max(abs(crossprod(x, residual)))) / lambda[i]
      primals[i] <- 0.5 * norm(residual, "2")^2 + lambda[i] * sum(abs(beta))
      duals[i] <-
        0.5 * norm(y, "2")^2 - 0.5 * norm(y - residual / dual_scale, "2")^2
    } else if (family == "binomial") {
      xbeta <- x %*% beta[, i]
      exp_xbeta <- exp(xbeta)
      pr <- exp_xbeta / (1 + exp_xbeta)
      pr <- ifelse(pr < 1e-5, 1e-5, pr)
      pr <- ifelse(pr > 1 - 1e-5, 1 - 1e-5, pr)

      residual <- y - pr

      correlation <- Matrix::crossprod(x, residual)

      theta <- residual / max(lambda, max(abs(correlation)))

      prx <- y - lambda * theta
      prx <- ifelse(prx < 1e-5, 1e-5, prx)
      prx <- ifelse(prx > 1 - 1e-5, 1 - 1e-5, prx)

      primals[i] <-
        -sum(y * xbeta - log1p(exp(xbeta))) + lambda * sum(abs(beta))
      duals[i] <- -sum(prx * log(prx) + (1 - prx) * log(1 - prx))
    }
  }

  tol_mod <- if (family == "gaussian") norm(y, "2")^2 else log(2) * length(y)

  list(
    primals = primals,
    duals = duals,
    gaps = primals - duals,
    rel_gaps = (primals - duals) / tol_mod,
    below_tol = primals - duals <= tol_gap * tol_mod
  )
}
