#pragma once

#include <algorithm>

template<typename T>
inline int
signum(T val)
{
  return (T(0) < val) - (val < T(0));
}

inline double
prox(const double x, const double lambda)
{
  return signum(x) * std::max(std::abs(x) - lambda, 0.0);
}
