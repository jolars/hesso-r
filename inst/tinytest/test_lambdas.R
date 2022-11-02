library(hesso)

n <- 100
p <- 1000

x <- matrix(rnorm(n * p), n, p)
beta <- rnorm(p)

y <- x %*% beta + rnorm(n)

family <- "gaussian"

res_glmnet <- glmnet::glmnet(x, y, intercept = FALSE, standardize = FALSE)
lambda <- res_glmnet$lambda * n
res_hesso <- lasso_path(x, y, family = family, lambda = lambda, standardize = FALSE)

compute_gaps <- function(x, y, betas, lambda) {
  path_length <- length(lambda)
  primals <- double(path_length)
  duals <- double(path_length)

  for (i in seq_len(path_length)) {
    beta <- betas[, i]
    r <- y - x %*% beta
    dual_scale <- max(1, norm(crossprod(x, r), "I") / lambda[i])
    primals[i] <- 0.5 * norm(r, "2")^2 + lambda[i] * sum(abs(beta))
    duals[i] <- 0.5 * norm(y, "2")^2 - 0.5 * norm(y - r / dual_scale, "2")^2
  }

  gaps <- primals - duals

  gaps
}

compute_gaps(x, y, res_glmnet$beta, lambda)
compute_gaps(x, y, res_hesso$beta, lambda)
