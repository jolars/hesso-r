library(hesso)

n <- 100
p <- 5

for (family in c("gaussian")) {
  d <- generate_design(n, p, family = family, density = 0.1)
  x <- d$x
  y <- d$y

  for (standardize in c(TRUE, FALSE)) {
    set.seed(1)
    fit_dense <- lasso(as.matrix(x), y)
    set.seed(1)
    fit_sparse <- lasso(x, y)

    expect_equal(fit_dense$lambda, fit_sparse$lambda)
    expect_equal(fit_dense$dev_ratio, fit_sparse$dev_ratio)
    expect_equal(fit_dense$beta, fit_sparse$beta)
  }
}

