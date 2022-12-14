#pragma once

#include "objective.hpp"
#include "subsetting.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

namespace hesso {

template<typename T, typename P>
void
updateHessian(Eigen::MatrixXd& H,
              Eigen::MatrixXd& Hinv,
              const T& x,
              const P& objective,
              const std::vector<size_t>& active,
              const std::vector<size_t>& active_new,
              const std::vector<size_t>& keep,
              const std::vector<size_t>& drop)
{
  using namespace Eigen;

  const size_t n = x.rows();

  if (!drop.empty()) {
    const MatrixXd Hinv_kd = subset(Hinv, keep, drop);
    const MatrixXd Hinv_kk = subset(Hinv, keep, keep);
    const MatrixXd Hinv_dd = subset(Hinv, drop, drop);

    Hinv = Hinv_kk - Hinv_kd * Hinv_dd.selfadjointView<Lower>().llt().solve(
                                 Hinv_kd.transpose());

    H = subset(H, keep, keep);
  }

  if (!active_new.empty()) {
    MatrixXd D = objective.hessian(x, active_new, active_new);
    MatrixXd B = objective.hessian(x, active, active_new);
    MatrixXd S = D - B.transpose() * Hinv.selfadjointView<Lower>() * B;

    double gamma = 1e-5 * n;
    double tau = 0;

    if (S.minCoeff() <= 0) {
      tau = -S.minCoeff() + gamma;
    }

    LLT<MatrixXd> llt(S.cols());

    double tau_old{ 0 };

    while (true) {
      llt.compute(S);

      if (llt.info() == Eigen::ComputationInfo::Success)
        break;

      tau = std::max(2 * tau, gamma);

      S.diagonal().array() += (tau - tau_old);

      tau_old = tau;
    }

    MatrixXd Sinv = llt.solve(MatrixXd::Identity(S.rows(), S.cols()));

    MatrixXd Hinv_B_Sinv = Hinv * B * Sinv;

    const size_t N = H.cols();
    const size_t M = D.cols();

    Hinv += Hinv_B_Sinv * B.transpose() * Hinv;

    MatrixXd H_new(N + M, N + M);

    H_new.topLeftCorner(N, N) = H;
    H_new.topRightCorner(N, M) = B;
    H_new.bottomLeftCorner(M, N) = B.transpose();
    H_new.bottomRightCorner(M, M) = D;

    MatrixXd Hinv_new(N + M, N + M);

    Hinv_new.topLeftCorner(N, N) = Hinv;
    Hinv_new.topRightCorner(N, M) = -Hinv_B_Sinv;
    Hinv_new.bottomLeftCorner(M, N) = -Hinv_B_Sinv.transpose();
    Hinv_new.bottomRightCorner(M, M) = Sinv;

    H = H_new;
    Hinv = Hinv_new;
  }
}

} // namespace hesso
