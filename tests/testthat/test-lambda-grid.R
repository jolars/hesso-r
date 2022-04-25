library(hesso)

test_that("lambda grid calculations are correct", {
  for (p in c(10, 100)) {
    set.seed(1110)

    n <- 50

    x <- matrix(rnorm(n * p), n, p)
    beta <- rnorm(p)
    y <- x %*% beta + rnorm(n)

    x <- scale(x)
    y <- y - mean(y)

    family <- "gaussian"

    res_work <- lassoPath(x, y, family = family)
    res_glmn <- glmnet::glmnet(x, y, intercept = FALSE)

    n_lambda <- min(length(res_work$lambda), length(res_glmn$lambda))

    expect_equal(res_work$lambda[1:n_lambda] / n, res_glmn$lambda[1:n_lambda])
  }
})
