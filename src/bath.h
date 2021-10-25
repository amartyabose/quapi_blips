#ifndef BATH_H
#define BATH_H

#include <cmath>
#include <complex>
#include <string>

#include <armadillo>

struct Bath {
    double beta;
    bool discrete;

    std::string type;
    double      omegac, xi, kappa;
    double      power;

    arma::cx_vec         etamn, eta0m, eta0e;
    std::complex<double> eta00, etamm;
    arma::mat            j_w;
    bool                 classical;

    void eta_coefficients(size_t n, double dt);
};

#endif
