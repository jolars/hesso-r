a <- double(path_length)
for (i in 0:(path_length - 1)) {
  a[i+1] <- lambda_max * lambda_min_ratio^((i)/(path_length - 1))
}
print(a)
