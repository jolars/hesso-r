#pragma once

#include <cmath>
#include <vector>

namespace hesso {

template<typename T>
int
signum(T val)
{
  return (T(0) < val) - (val < T(0));
}

template<typename T, typename N>
std::vector<T>
geomSpace(const T from, const T to, const N n)
{
  std::vector<T> x;
  x.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    x.emplace_back(from *
                   std::pow(to / from, static_cast<double>(i) / (n - 1)));
  }

  return x;
}

}
