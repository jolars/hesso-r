#pragma once

#include <algorithm>
#include <vector>

template<typename T, typename S>
inline bool
contains(const T& x, const S& what)
{
  return std::find(x.begin(), x.end(), what) != x.end();
}

inline std::vector<int>
setUnion(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  out.reserve(a.size() + b.size());

  std::set_union(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  out.shrink_to_fit();

  return out;
}

inline std::vector<int>
setDiff(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  out.reserve(a.size());

  std::set_difference(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  out.shrink_to_fit();
  return out;
}

inline std::vector<int>
setIntersect(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  out.reserve(std::min(a.size(), b.size()));

  std::set_intersection(
    a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));

  out.shrink_to_fit();

  return out;
}

// set intersection that retains permutation in `a`
inline std::vector<int>
safeSetIntersect(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  out.reserve(std::min(a.size(), b.size()));

  for (auto&& a_i : a) {
    if (contains(b, a_i)) {
      out.emplace_back(a_i);
    }
  }

  out.shrink_to_fit();
  return out;
}

// set difference that retains permutation in `a`
inline std::vector<int>
safeSetDiff(const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> out;
  out.reserve(a.size());

  for (auto&& a_i : a) {
    if (!contains(b, a_i)) {
      out.emplace_back(a_i);
    }
  }

  out.shrink_to_fit();
  return out;
}
