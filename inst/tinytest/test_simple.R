library(hesso)
library(glmnet)

set.seed(53)

n <- 10
p <- 11
family <- "gaussian"
density <- 1
standardize <- FALSE
intercept <- FALSE
tol <- 1e-8
verbosity <- 0

data <- generate_design(
  n,
  p,
  family = family,
  density = density
)

popsd <- function(x) {
  sd(x) * sqrt((length(x) - 1) / length(x))
}

x <- data$x
y <- data$y 

fit_glmnet <- glmnet(
  x,
  y,
  family,
  intercept = intercept,
  # lambda = lambda / n,
  standardize = standardize,
  thresh = 1e-10
)

lambda <- fit_glmnet$lambda * n

fit <- lasso(
  x,
  y,
  lambda = lambda,
  tol = tol,
  max_it = 1500,
  warm_starts = TRUE
)

print(fit_glmnet$beta)
print(fit$beta)

gaps <- duals <- primals <- double(length(lambda))

for (i in 1:length(primals)) {
  b <- fit$beta[, i]

  residual <- x %*% b - y
  gradient <- t(x) %*% residual
  theta <- residual / max(1, norm(gradient, "I") / lambda[i])

  primals[i] <- 0.5 * norm(residual, "2")^2 + lambda[i] * sum(abs(b))
  duals[i] <- 0.5 * norm(y, "2")^2 - 0.5 * norm(theta + y, "2")^2

  gaps[i] <- primals[i] - duals[i]
}
print(gaps)
print(fit$passes)

stopifnot(all(gaps < tol))

