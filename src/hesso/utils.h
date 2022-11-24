#pragma once

#include <Rcpp.h>
#include <vector>

template<typename T>
void
print(const std::vector<T>& x, const std::string& what) {
  Rcpp::Rcout << what << ":" << std::endl;
  for (const auto& x_i : x) {
    Rcpp::Rcout << " " << x_i;
  }
  Rcpp::Rcout << std::endl;
}
