#include "bath.h"

const double               pi = std::acos(-1.);
const std::complex<double> I(0, 1);

std::complex<double> trapz(arma::vec x, arma::cx_vec vals,
                           bool discrete = false) {
    if (discrete)
        return arma::sum(vals);

    std::complex<double> val = 0; // (vals(0) + vals(vals.n_rows-1))/2.;
    for (size_t i = 1; i < vals.n_rows; i++)
        val += (vals(i) + vals(i - 1)) / 2.;

    return val * (x(1) - x(0));
}

void Bath::eta_coefficients(size_t n, double dt) {
    arma::vec w, spect;
    if (j_w.n_rows == 0) {
        if (type == "ohmic") {
            w     = arma::linspace(-30 * omegac, 30 * omegac, 200000);
            spect = pi / 2 * xi * std::pow(omegac, 1-power) * arma::pow(arma::abs(w), power) % arma::exp(-arma::abs(w) / omegac) % arma::sign(w);
        } else if (type == "drude") {
            w     = arma::linspace(-30 * omegac, 30 * omegac, 200000);
            spect = 2 * kappa * std::pow(omegac, 2-power) * arma::pow(arma::abs(w), power) / (w % w + omegac*omegac) % arma::sign(w);
        }
    } else {
        w     = j_w.col(0);
        spect = j_w.col(1);
    }
    arma::vec common_part;
    if (classical)
        common_part = spect / (w % w) % (2. / (w * beta) + 1.);
    else if (beta < 0)
        common_part = spect / (w % w) * 2.0;
    else
        common_part = spect / (w % w) % (2.0 / (1.0 - arma::exp(-w * beta)));

    eta00 = 1. / (2 * pi) *
            trapz(w, common_part % (1 - arma::exp(-I * w * dt / 2.)), discrete);
    etamm = 1. / (2 * pi) *
            trapz(w, common_part % (1 - arma::exp(-I * w * dt)), discrete);
    eta0m = arma::zeros<arma::cx_vec>(n);
    eta0e = arma::zeros<arma::cx_vec>(n);
    etamn = arma::zeros<arma::cx_vec>(n);

    for (size_t k = 1; k <= n; k++) {
        eta0m(k - 1) = 2. / pi *
                       trapz(w,
                             common_part % arma::sin(w * dt / 4.) %
                                 arma::sin(w * dt / 2.) %
                                 arma::exp(-I * w * (k - 0.25) * dt),
                             discrete);
        eta0e(k - 1) = 2. / pi *
                       trapz(w,
                             common_part % arma::sin(w * dt / 4.) %
                                 arma::sin(w * dt / 4.) %
                                 arma::exp(-I * w * (k - 0.5) * dt),
                             discrete);
        etamn(k - 1) =
            2. / pi *
            trapz(w,
                  common_part % arma::sin(w * dt / 2.) %
                      arma::sin(w * dt / 2.) % arma::exp(-I * w * k * dt),
                  discrete);
    }
}
