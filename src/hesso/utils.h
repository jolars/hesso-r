#pragma once

#include <Rcpp.h>
#include <algorithm>
#include <vector>

template<typename T>
void
print(const std::vector<T>& x, const std::string& what)
{
  Rcpp::Rcout << what << ":" << std::endl;
  for (const auto& x_i : x) {
    Rcpp::Rcout << " " << x_i;
  }
  Rcpp::Rcout << std::endl;
}

template<typename T, typename S>
bool
contains(const T& x, const S& what)
{
  return std::find(x.cbegin(), x.cend(), what) != x.end();
}
