library(hesso)
library(glmnet)

grid <- expand.grid(
  # np = list(c(100, 5), c(50, 200)),
  np = list(c(100, 5)),
  density = c(0.5, 1),
  family = c("gaussian", "binomial"),
  standardize = c(FALSE, TRUE),
  stringsAsFactors = FALSE
)

tol_gap <- 1e-6

for (i in seq_len(nrow(grid))) {
  set.seed(i)

  g <- grid[i, ]

  np <- g$np[[1]]
  n <- np[1]
  p <- np[2]
  family <- g$family
  standardize <- g$standardize
  density <- g$density
  intercept <- g$intercept

  data <- generate_design(
    n,
    p,
    family = g$family,
    density = g$density
  )

  x <- data$x
  y <- data$y

  # if (family == "gaussian") {
  #   y <- y - mean(y)
  # }

  fit <- lasso_path(
    x,
    y,
    family,
    standardize = standardize,
    tol_gap = tol_gap,
    store_dual_variables = TRUE
  )

  fit_glmnet <- glmnet(
    x,
    y,
    family,
    intercept = FALSE,
    # lambda = fit$lambda / length(y),
    standardize = standardize,
    thresh = 1e-20
  )

  # gaps <- check_gaps(fit, standardize, x, y, tol_gap)
  # check_gaps(fit$beta, fit$lambda, x, y, standardize, tol_gap)

  expect_equal(fit$dev_ratio, fit_glmnet$dev.ratio, tolerance = 1e-5)

  # expect_true(all(gaps$below_tol))
}
