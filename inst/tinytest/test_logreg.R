library(hesso)

datalist <- "leukemia"

tol_gap <- 1e-4
standardize <- FALSE

for (dataset in datalist) {
  data(list = c(dataset))
  d <- get(dataset)
  x <- d$x
  y <- d$y

  fit <- lasso_path(
    x,
    y,
    "binomial",
    standardize = standardize,
    tol_gap = tol_gap,
    verbosity = 0,
    store_dual_variables = TRUE
  )

  gaps <- check_gaps(fit, standardize, x, y, tol_gap)

  expect_true(all(gaps$below_tol))
}
