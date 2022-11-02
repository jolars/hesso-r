library(hesso)

n <- 100
p <- 5

for (family in c("gaussian", "binomial")) {
  d <- generate_design(n, p, family = family, density = 0.5)
  x <- d$x
  y <- d$y

  for (standardize in c(TRUE, FALSE)) {
    set.seed(1)
    fit_dense <- lasso_path(
      as.matrix(x),
      y,
      family,
      standardize = standardize
    )
    set.seed(1)
    fit_sparse <- lasso_path(
      x,
      y,
      family,
      standardize = standardize
    )

    expect_equal(fit_dense$lambda, fit_sparse$lambda)
    expect_equal(fit_dense$dev_ratio, fit_sparse$dev_ratio)
    expect_equal(fit_dense$beta, fit_sparse$beta)
  }
}

