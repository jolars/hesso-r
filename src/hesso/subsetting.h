#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <vector>

template<typename T1, typename T2>
std::vector<T1>
subset(const std::vector<T1>& x, const std::vector<T2>& indices)
{
  std::vector<T1> out(indices.size(), 0);

  std::transform(indices.begin(), indices.end(), out.begin(), [x](T2 pos) {
    return x[pos];
  });

  return out;
}

template<typename T>
Eigen::MatrixXd
subset(const Eigen::MatrixXd& m,
       const std::vector<T>& rows,
       const std::vector<T>& cols)
{
  Eigen::MatrixXd m_subset(rows.size(), cols.size());

  for (size_t i = 0; i < rows.size(); ++i) {
    for (size_t j = 0; j < cols.size(); ++j) {
      m_subset(i, j) = m(rows[i], cols[j]);
    }
  }

  return m_subset;
}
