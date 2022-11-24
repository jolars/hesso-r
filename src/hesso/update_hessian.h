#pragma once

#include "objective.h"
#include "set_operations.h"
#include "subsetting.h"
// #include <Eigen/Core>
// #include <Eigen/Eigenvalues>
#include <RcppEigen.h>

template<typename T, typename P>
void
updateHessian(Eigen::MatrixXd& H,
              Eigen::MatrixXd& Hinv,
              const T& x,
              P& objective,
              const std::vector<size_t>& active,
              const std::vector<size_t>& active_new,
              const std::vector<size_t>& keep,
              const std::vector<size_t>& drop,
              const bool verify_hessian,
              const bool verbose)
{
  using namespace Eigen;

  const size_t n = x.rows();

  if (!drop.empty()) {
    if (verbose) {
      Rprintf("    dropping deactivated predictors for inverse (n = %i)\n",
              drop.size());
    }

    const MatrixXd Hinv_kd = subset(Hinv, keep, drop);
    const MatrixXd Hinv_kk = subset(Hinv, keep, keep);
    const MatrixXd Hinv_dd = subset(Hinv, drop, drop);

    Hinv = Hinv_kk - Hinv_kd * Hinv_dd.selfadjointView<Upper>().llt().solve(
                                 Hinv_kd.transpose());

    H = subset(H, keep, keep);
  }

  if (!active_new.empty()) {
    if (verbose) {
      Rprintf("    adding newly activated predictors to inverse (n = %i)\n",
              active_new.size());
    }

    MatrixXd D = objective.hessian(x, active_new, active_new);
    MatrixXd B = objective.hessian(x, active, active_new);
    MatrixXd S = D - B.transpose() * Hinv * B;

    double gamma = 1e-5 * n;
    double tau = 0;

    if (S.minCoeff() <= 0) {
      tau = -S.minCoeff() + gamma;
    }

    LLT<MatrixXd> llt(S.cols());

    double tau_old {0};

    while (llt.info() != Eigen::ComputationInfo::Success) {
      Rcpp::Rcout << "trying LLT\n";
      llt.compute(S);

      tau = std::max(2 * tau, gamma);

      S.diagonal().array() += (tau - tau_old);

      tau_old = tau;
    }

    MatrixXd I = MatrixXd::Identity(S.rows(), S.cols());
    MatrixXd Sinv = llt.solve(I);

    // SelfAdjointEigenSolver<MatrixXd> eigen_decomp(S);

    // MatrixXd Q = eigen_decomp.eigenvectors();
    // VectorXd l = eigen_decomp.eigenvalues();

    // if (l.minCoeff() < 1e-4 * n) {
    //   for (long int i = 0; i < l.size(); ++i) {
    //     D(i, i) += 1e-4 * n;
    //     l(i) += 1e-4 * n;
    //   }
    // }

    // VectorXd l_inv(l.size());
    // for (long int i = 0; i < l.size(); ++i) {
    //   l_inv(i) = 1.0 / l(i);
    // }

    // MatrixXd Sinv = Q * l_inv.asDiagonal() * Q.transpose();

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

  if (verify_hessian) {
    double hess_inv_error = (H - H * Hinv * H).lpNorm<Eigen::Infinity>();

    if (hess_inv_error >= 1e-2) {
      Rcpp::stop("inverse matrix computation is incorrect");
    }
  }
}
