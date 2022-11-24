library(hesso)

set.seed(1)

x <- matrix(rnorm(25), 5)

active_perm_prev <- c(1, 3, 2)
active_set_prev <- sort(active_perm_prev)
active_perm <- c(1, 3, 4)
active_set <- sort(active_perm)

h <- t(x[, active_perm_prev]) %*% x[, active_perm_prev]
h_inv <- solve(h)

verify_hessian <- TRUE
verbose <- TRUE

hesso:::update_hessian(
  h,
  h_inv,
  x,
  active_set - 1,
  active_set_prev - 1,
  active_perm - 1,
  active_perm_prev - 1,
  verify_hessian,
  verbose
)

true_h <- t(x[, active_perm]) %*% x[, active_perm]
print(true_h)
solve(true_h)
