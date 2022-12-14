library(hesso)
library(glmnet)

set.seed(51)

n <- 100
p <- 10
family <- "gaussian"
density <- 1
standardize <- FALSE
intercept <- TRUE
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
lambda <- lambda[1:5]

fit <- lasso(
  x,
  y,
  intercept = intercept,
  # lambda = NULL,
  tol = tol,
  max_it = 100000,
  warm_starts = TRUE
)

# print(fit_glmnet$beta)
# print(fit$beta)

# print(fit_glmnet$a0)
# print(fit$beta0)

gaps <- duals <- primals <- double(length(lambda))

beta <- fit$beta
beta0 <- fit$beta0

# beta <- fit_glmnet$beta
# beta0 <- fit_glmnet$a0

for (i in 1:length(lambda)) {
  b <- beta[, i]

  residual <- x %*% b + beta0[i]*intercept - y
  gradient <- t(x) %*% residual
  # dual_scaling <- max(1, norm(gradient, "I") / lambda[i])
  dual_scaling <- max(1.0, norm(gradient, "I") / lambda[i])
  theta <- residual / dual_scaling

  primals[i] <- 0.5 * norm(residual, "2")^2 + lambda[i] * sum(abs(b))
  duals[i] <- 0.5 * (norm(y, "2")^2 - norm(theta + y, "2")^2)

  gaps[i] <- primals[i] - duals[i]
}

tinytest::expect_true(all(gaps <= tol))

