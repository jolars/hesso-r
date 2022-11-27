library(hesso)
library(glmnet)

set.seed(51)

n <- 100
p <- 5000
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

lambda <- lambda

fit <- lasso(
  x,
  y,
  lambda = NULL,
  tol = tol,
  max_it = 100000,
  warm_starts = TRUE
)

# print(fit_glmnet$beta)
# print(fit$beta)

print(lambda)
print(fit$lambda)

# gaps <- duals <- primals <- double(length(lambda))

# for (i in 1:length(primals)) {
#   b <- fit$beta[, i]

#   residual <- x %*% b - y
#   gradient <- t(x) %*% residual
#   theta <- residual / max(1, norm(gradient, "I") / lambda[i])

#   primals[i] <- 0.5 * norm(residual, "2")^2 + lambda[i] * sum(abs(b))
#   duals[i] <- 0.5 * norm(y, "2")^2 - 0.5 * norm(theta + y, "2")^2

#   gaps[i] <- primals[i] - duals[i]
# }

# print(fit$passes)
# print(fit$gaps)

# # ind <- fit$beta[, 2] != 0
# # print(fit$beta[ind, ])

# # print(which(fit$beta[, 2] != 0, 2))
# # print(fit$beta[fit$beta[, 2] != 0, 2])
# # print(fit_glmnet$beta[fit_glmnet$beta[, 2] != 0, 2])

# if (!all(gaps <= tol)) {
#   print(gaps)
#   stop("did not converge")
# }

