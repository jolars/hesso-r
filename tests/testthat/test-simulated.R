test_that("gaussian and logistic models for simulated data", {
  library(hesso)

  grid <- expand.grid(
    np = list(c(100, 5), c(50, 200)),
    density = c(0.5, 1),
    family = c("gaussian", "binomial"),
    standardize = c(FALSE),
    stringsAsFactors = FALSE
  )

  tol_gap <- 1e-4

  for (i in seq_len(nrow(grid))) {
    set.seed(i)

    g <- grid[i, ]

    np <- g$np[[1]]
    n <- np[1]
    p <- np[2]
    family <- g$family
    standardize <- g$standardize
    density <- g$density

    data <- generate_design(
      n,
      p,
      family = g$family,
      density = g$density
    )

    x <- data$x
    y <- data$y

    if (family == "gaussian") {
      y <- y - mean(y)
    }

    fit <- lassoPath(
      x,
      y,
      family,
      verbosity = 0,
      standardize = standardize,
      tol_gap = tol_gap,
      store_dual_variables = TRUE
    )

    gaps <- check_gaps(fit, standardize, x, y, tol_gap)

    expect_true(all(gaps$below_tol))
  }
})
