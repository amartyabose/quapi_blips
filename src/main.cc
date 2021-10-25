#include "bath.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#pragma omp declare reduction(+ : arma::mat : omp_out += omp_in) initializer (omp_priv = omp_orig)
#pragma omp declare reduction(+ : arma::vec : omp_out += omp_in) initializer (omp_priv = omp_orig)
#pragma omp declare reduction(+ : arma::cx_mat : omp_out += omp_in) initializer (omp_priv = omp_orig)
#pragma omp declare reduction(+ : arma::cx_vec : omp_out += omp_in) initializer (omp_priv = omp_orig)

struct Path;
#pragma omp declare reduction (merge : std::vector<Path> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (+ : std::vector<arma::cx_mat> : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<>{}))\
    initializer (omp_priv = omp_orig)

#include <fmt/core.h>

#include <boost/program_options.hpp>
namespace opt = boost::program_options;

std::complex<double> const I(0, 1);

size_t ndims;
size_t N, Kmax, num_blips;
double dt;

arma::cx_mat H;
arma::cx_mat U;
arma::cx_mat Udag;
arma::cx_mat Q0;
arma::cx_vec rho0;

std::vector<std::pair<size_t, size_t>> blip_states;

arma::vec blip_vals;
arma::vec avg_vals_for_blips;
arma::vec sojourn_vals;
unsigned  base;

Bath bath;

arma::cx_mat expmat_with_i(arma::cx_mat const &A) {
    arma::cx_mat vec;
    arma::vec    val;
    arma::eig_sym(val, vec, A);
    arma::cx_mat a = arma::zeros<arma::cx_mat>(A.n_rows, A.n_cols);
    for (unsigned i = 0; i < vec.n_rows; i++)
        a(i, i) = std::exp(I * val(i));
    return vec * a * vec.t();
}

arma::mat convert_j_over_w_to_j(arma::mat const &j_over_w) {
    arma::mat j_temp = j_over_w;
    j_temp.col(1)    = j_over_w.col(1) % j_over_w.col(0);

    if (j_over_w(0, 0) != 0)
        return arma::join_cols(-arma::flipud(j_temp), j_temp);

    arma::mat average = arma::zeros<arma::mat>(j_temp.n_rows - 1, 2);
    for (unsigned r = 0; r < j_temp.n_rows - 1; r++)
        average.row(r) = (j_temp.row(r) + j_temp.row(r + 1)) / 2;

    arma::mat j = arma::join_cols(-arma::flipud(average), average);

    return j;
}

arma::mat convert_freq_coupl_to_j(arma::mat const &freq_coupl) {
    arma::mat temp = freq_coupl;
    temp.col(1) = arma::datum::pi / 2 * freq_coupl.col(1) % freq_coupl.col(1) /
                  freq_coupl.col(0);
    return arma::join_cols(-arma::flipud(temp), temp);
}

void parse_param(char const *filename) {
    std::string spect, freq_coupl, hamiltonian, rho0_string, dvr_string;

    opt::options_description desc("Options required in the configuration file");
    desc.add_options()("BATH.omegac", opt::value<double>(&bath.omegac),
                       "bath cutoff frequency");
    desc.add_options()("BATH.xi", opt::value<double>(&bath.xi),
                       "Kondo parameter");
    desc.add_options()("BATH.power",
                       opt::value<double>(&bath.power)->default_value(1),
                       "Power of the rise part");
    desc.add_options()("BATH.kappa", opt::value<double>(&bath.kappa),
                       "reorganization energy Drude bath");
    desc.add_options()(
        "BATH.type",
        opt::value<std::string>(&bath.type)->default_value("ohmic"),
        "type of bath");
    desc.add_options()("BATH.beta", opt::value<double>(&bath.beta),
                       "inverse temperature");
    desc.add_options()("BATH.classical",
                       opt::value<bool>(&bath.classical)->default_value(false),
                       "is it a classical Boltzmann bath?");
    desc.add_options()("BATH.j_w_over_w", opt::value<std::string>(&spect),
                       "J(w)/w");
    desc.add_options()("BATH.freq_coupl", opt::value<std::string>(&freq_coupl)->default_value(""),
                       "frequency-coupling");
    desc.add_options()("SIMULATION.dt", opt::value<double>(&dt),
                       "quantum time step");
    desc.add_options()("SIMULATION.N", opt::value<size_t>(&N),
                       "number of quantum time steps");
    desc.add_options()("SIMULATION.Kmax", opt::value<size_t>(&Kmax),
                       "number of quantum time steps; Kmax = "
                       "-1 means full memory "
                       "calculation");
    desc.add_options()(
        "SYSTEM.rho0",
        opt::value<std::string>(&rho0_string)->default_value("1 0 0 0"),
        "rho(0)");
    desc.add_options()("SYSTEM.blips",
                       opt::value<size_t>(&num_blips)->default_value(-1),
                       "max number of blips allowed. "
                       "-1 means all blips allowed");
    desc.add_options()(
        "SYSTEM.Hamiltonian",
        opt::value<std::string>(&hamiltonian)->default_value("0 -1; -1 0"),
        "Hamiltonian");
    desc.add_options()(
        "SYSTEM.dvrs",
        opt::value<std::string>(&dvr_string)->default_value("1 -1"),
        "DVR positions");

    if (std::string(filename) == "help" || std::string(filename) == "") {
        std::cerr << desc;
        std::exit(1);
    }

    if (freq_coupl != "" && spect != "") {
        std::cerr << desc;
        std::exit(1);
    }

    if (freq_coupl != "")
        bath.discrete = true;

    opt::variables_map vm;
    std::ifstream      ifs(filename);
    try {
        opt::store(opt::parse_config_file<char>(ifs, desc), vm);
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; }
    opt::notify(vm);

    sojourn_vals       = arma::vec(dvr_string);
    ndims              = sojourn_vals.n_rows;
    base               = ndims * ndims - ndims + 1;
    blip_vals          = arma::zeros<arma::vec>(base);
    avg_vals_for_blips = arma::zeros<arma::vec>(base);

    unsigned index = 1;
    for (unsigned i = 0; i < ndims; ++i)
        for (unsigned j = 0; j < ndims; ++j)
            if (i != j) {
                blip_vals(index) = sojourn_vals[i] - sojourn_vals[j];
                avg_vals_for_blips(index) =
                    (sojourn_vals[i] + sojourn_vals[j]) / 2;
                blip_states.push_back(std::make_pair(i, j));
                index++;
            }

    H    = arma::cx_mat(hamiltonian);
    rho0 = arma::cx_vec(rho0_string);

    U              = expmat_with_i(-H * dt);
    Udag           = U.t();
    Q0             = arma::zeros<arma::cx_mat>(ndims * ndims, ndims * ndims);
    arma::cx_mat Q = arma::kron(U, Udag);
    Q0(0, 0)       = Q(0, 0);
    Q0(0, 3)       = Q(0, 3);
    Q0(3, 0)       = Q(3, 0);
    Q0(3, 3)       = Q(3, 3);

    if (spect != "") {
        arma::mat j_w_over_w;
        j_w_over_w.load(spect);
        bath.j_w = convert_j_over_w_to_j(j_w_over_w);
    }

    if (freq_coupl != "") {
        arma::mat freq_coupls;
        freq_coupls.load(freq_coupl);
        freq_coupls.print();
        bath.j_w = convert_freq_coupl_to_j(freq_coupls);
    }

    if (Kmax > N)
        Kmax = N + 2; // Arbitrary small value for Kmax that allows full memory
                      // calculations

    bath.eta_coefficients(Kmax, dt);

    std::cout << "Setup done!" << std::endl;
}

struct Path {
    std::vector<std::pair<size_t, int>> reversed_blip_locations;

    arma::cx_mat amplitude, Tlast_term, Tlast_cont;
    std::string  path_string;
    Path(int b) {
        reversed_blip_locations.reserve(Kmax + 1);
        path_string = fmt::format("{}", b);
        amplitude   = arma::zeros<arma::cx_mat>(ndims * ndims, ndims * ndims);
        Tlast_term  = arma::zeros<arma::cx_mat>(ndims * ndims, ndims * ndims);
        Tlast_cont  = arma::zeros<arma::cx_mat>(ndims * ndims, ndims * ndims);
        if (!b)
            for (size_t i = 0; i < ndims; i++)
                amplitude(i * ndims + i, i * ndims + i) = 1;
        else {
            amplitude(b, b) = 1;
            reversed_blip_locations.push_back(std::make_pair(0, b));
            amplitude *=
                std::exp(-(bath.eta00.real() * blip_vals(b) +
                           2. * I * bath.eta00.imag() * avg_vals_for_blips(b)) *
                         blip_vals(b));
        }
    }

    Path()              = delete;
    Path(Path const &p) = default;
    Path(Path &&p)      = default;

    Path &operator=(Path const &p) = default;
    Path &operator=(Path &&p) = default;

    ~Path() = default;

    arma::cx_mat append(int const b) {
        path_string += fmt::format("{}", b);
        if (!b && path_string[path_string.size() - 2] == '0')
            return insert_sojourn_before_sojourn();
        else
            return insert_blip_before_sojourn(b);
    }

private:
    arma::cx_mat insert_sojourn_before_sojourn() {
        std::complex<double> w_term = 0, w_cont = 0;
        std::complex<double> w_Tlast_term = 0, w_Tlast_cont = 0;
        auto const           pos = path_string.size() - 1;
        for (auto bl : reversed_blip_locations) {
            if (bl.first == 0) {
                w_term -= 2 * blip_vals(bl.second) *
                          bath.eta0e(pos - bl.first - 1).imag();
                w_cont -= 2 * blip_vals(bl.second) *
                          bath.eta0m(pos - bl.first - 1).imag();
            } else {
                w_term -= 2 * blip_vals(bl.second) *
                          bath.eta0m(pos - bl.first - 1).imag();
                w_cont -= 2 * blip_vals(bl.second) *
                          bath.etamn(pos - bl.first - 1).imag();
            }
            w_Tlast_term -= 2 * blip_vals(bl.second) *
                            bath.eta0m(pos - bl.first - 1).imag();
            w_Tlast_cont -= 2 * blip_vals(bl.second) *
                            bath.etamn(pos - bl.first - 1).imag();
        }
        arma::cx_mat Q_term = Q0, Q_cont = Q0;

        Tlast_term = Q0;
        Tlast_cont = Q0;
        for (size_t c = 0; c < ndims; c++) {
            Q_term.col(c * ndims + c) *= std::exp(I * w_term * sojourn_vals(c));
            Q_cont.col(c * ndims + c) *= std::exp(I * w_cont * sojourn_vals(c));
            Tlast_term.col(c * ndims + c) *=
                std::exp(I * w_Tlast_term * sojourn_vals(c));
            Tlast_cont.col(c * ndims + c) *=
                std::exp(I * w_Tlast_cont * sojourn_vals(c));
        }
        arma::cx_mat const terminal_amplitude(amplitude * Q_term);
        amplitude = amplitude * Q_cont;
        return terminal_amplitude;
    }
    arma::cx_mat insert_blip_before_sojourn(int const b) {
        arma::cx_mat transfer_matrix(ndims * ndims, ndims * ndims);
        auto const [sp, sm]   = blip_states[b - 1];
        auto const last_point = path_string[path_string.size() - 2] - '0';

        if (last_point != 0) {
            auto const [sp_old, sm_old] = blip_states[last_point - 1];
            if (!b)
                for (size_t c = 0; c < ndims; c++)
                    transfer_matrix(last_point, c * ndims + c) =
                        U(sp_old, c) * Udag(c, sm_old);
            else
                transfer_matrix(last_point, b) =
                    U(sp_old, sp) * Udag(sm, sm_old);
        } else
            for (size_t r = 0; r < ndims; r++)
                transfer_matrix(r * ndims + r, b) = U(r, sp) * Udag(sm, r);

        auto const pos = path_string.size() - 1;

        std::complex<double> two_time_interaction_cont = 0,
                             two_time_interaction_term = 0;
        std::complex<double> w_Tlast_term = 0, w_Tlast_cont = 0;
        for (auto bl : reversed_blip_locations) {
            if (bl.first == 0) {
                two_time_interaction_term -=
                    blip_vals(bl.second) * bath.eta0e(pos - bl.first - 1);
                two_time_interaction_cont -=
                    blip_vals(bl.second) * bath.eta0m(pos - bl.first - 1);
            } else {
                two_time_interaction_term -=
                    blip_vals(bl.second) * bath.eta0m(pos - bl.first - 1);
                two_time_interaction_cont -=
                    blip_vals(bl.second) * bath.etamn(pos - bl.first - 1);
            }
            w_Tlast_term -=
                blip_vals(bl.second) * bath.eta0m(pos - bl.first - 1);
            w_Tlast_cont -=
                blip_vals(bl.second) * bath.etamn(pos - bl.first - 1);
        }

        arma::cx_mat terminal_amplitude(ndims * ndims, ndims * ndims);
        if (b) {
            auto self_interaction_cont =
                -(bath.etamm.real() * blip_vals(b) +
                  2. * I * bath.etamm.imag() * avg_vals_for_blips(b)) *
                blip_vals(b);
            auto self_interaction_term =
                -(bath.eta00.real() * blip_vals(b) +
                  2. * I * bath.eta00.imag() * avg_vals_for_blips(b)) *
                blip_vals(b);

            Tlast_cont =
                transfer_matrix *
                std::exp(self_interaction_cont +
                         w_Tlast_cont.real() * blip_vals(b) +
                         2. * I * w_Tlast_cont.imag() * avg_vals_for_blips(b));
            Tlast_term =
                transfer_matrix *
                std::exp(self_interaction_term +
                         w_Tlast_term.real() * blip_vals(b) +
                         2. * I * w_Tlast_term.imag() * avg_vals_for_blips(b));

            terminal_amplitude =
                amplitude * transfer_matrix *
                std::exp(self_interaction_term +
                         two_time_interaction_term.real() * blip_vals(b) +
                         2. * I * two_time_interaction_term.imag() *
                             avg_vals_for_blips(b));
            amplitude *=
                transfer_matrix *
                std::exp(self_interaction_cont +
                         two_time_interaction_cont.real() * blip_vals(b) +
                         2. * I * two_time_interaction_cont.imag() *
                             avg_vals_for_blips(b));

            reversed_blip_locations.push_back(std::make_pair(pos, b));
        } else {
            arma::cx_mat terminal_transfer = transfer_matrix;

            Tlast_term = transfer_matrix;
            Tlast_cont = transfer_matrix;
            for (size_t c = 0; c < ndims; c++) {
                auto term_val =
                    std::exp(2. * I * two_time_interaction_term.imag() *
                             sojourn_vals(c));
                auto cont_val =
                    std::exp(2. * I * two_time_interaction_cont.imag() *
                             sojourn_vals(c));
                terminal_transfer.col(c * ndims + c) *= term_val;
                transfer_matrix.col(c * ndims + c) *= cont_val;
                Tlast_term.col(c * ndims + c) *= term_val;
                Tlast_cont.col(c * ndims + c) *= cont_val;
            }
            terminal_amplitude = amplitude * terminal_transfer;
            amplitude *= transfer_matrix;
        }
        return terminal_amplitude;
    }
};

arma::cx_vec extend_paths_within_memory(std::vector<Path> &paths,
                                        size_t const       num_blips) {
    std::vector<Path> new_paths;
    new_paths.reserve(paths.size());
    arma::cx_vec rho = arma::zeros<arma::cx_vec>(rho0.n_rows);
#pragma omp parallel for reduction(+ : rho) reduction(merge : new_paths)
    for (auto const &p : paths) {
        Path         temp_p  = p;
        arma::cx_vec del_rho = temp_p.append(0) * rho0;
        rho += del_rho;
        new_paths.push_back(temp_p);

        if (p.reversed_blip_locations.size() < num_blips)
            for (unsigned bn = 1; bn < base; ++bn) {
                temp_p  = p;
                del_rho = temp_p.append(bn) * rho0;
                rho += del_rho;
                new_paths.push_back(temp_p);
            }
    }

    paths = std::move(new_paths);
    return rho;
}

unsigned hash_path_string(std::string const &str) {
    unsigned factor = 1;
    unsigned number = 0;
    for (char const &c : str) {
        number += (c - '0') * factor;
        factor *= base;
    }

    return number;
}

arma::cx_vec extend_paths_beyond_memory(std::vector<Path> &paths) {
    auto const arr_length = paths.size() / base;

    std::vector<arma::cx_mat> transfer_mat_cont(
        arr_length, arma::zeros<arma::cx_mat>(ndims * ndims, ndims * ndims));
    std::vector<arma::cx_mat> transfer_mat_term(
        arr_length, arma::zeros<arma::cx_mat>(ndims * ndims, ndims * ndims));

#pragma omp parallel for reduction(+:transfer_mat_cont) reduction(+:transfer_mat_term)
    for (auto const &p : paths) {
        auto const truncated_path_string =
            p.path_string.substr(0, p.path_string.size() - 1);
        transfer_mat_cont[hash_path_string(truncated_path_string)] +=
            p.Tlast_cont;
        transfer_mat_term[hash_path_string(truncated_path_string)] +=
            p.Tlast_term;
    }

    arma::cx_vec rho = arma::zeros<arma::cx_vec>(rho0.n_rows);
#pragma omp parallel for reduction(+ : rho)
    for (auto &p : paths) {
        auto const truncated_path_string =
            p.path_string.substr(1, p.path_string.size() - 1);
        rho += p.amplitude *
               transfer_mat_term[hash_path_string(truncated_path_string)] *
               rho0;
        p.Tlast_term =
            p.Tlast_cont *
            transfer_mat_term[hash_path_string(truncated_path_string)];
    }

    return rho;
}

arma::cx_vec extend_paths(std::vector<Path> &paths, size_t const num_blips,
                          size_t const step_num) {
    if (step_num <= Kmax)
        return extend_paths_within_memory(paths, num_blips);
    else
        return extend_paths_beyond_memory(paths);
}

int main(int argc, char **argv) {
    parse_param(argv[1]);

    std::cout << bath.eta00 << "\t" << bath.etamm << std::endl;
    bath.eta0m.print("eta0m");
    bath.eta0e.print("eta0e");
    bath.etamn.print("etamn");

    std::vector<Path> paths;
    paths.push_back(Path(0));
    if (num_blips) {
        paths.push_back(Path(1));
        paths.push_back(Path(2));
    }

    auto       rho = rho0;
    std::FILE *fp  = std::fopen((std::string(argv[1]) + ".out").c_str(), "w");
    fmt::print(fp,
               "{:^8s}\t{:^23s}\t{:^23s}\t{:^23s}\t{:^23s}\t{:^23s}\t{:^23s}\t{"
               ":^23s}\t{:^23s}\t{:^s}\n",
               "time", "Re(rho11)", "Im(rho11)", "Re(rho12)", "Im(rho12)",
               "Re(rho21)", "Im(rho21)", "Re(rho22)", "Im(rho22)", "#paths");
    fmt::print(fp,
               "{:^8f}\t{:+.15e}\t{:+.15e}\t{:+.15e}\t{:+.15e}\t{:+.15e}\t{:+."
               "15e}\t{:+.15e}\t{:+.15e}\t{:6d}\n",
               (0 * dt), rho(0).real(), rho(0).imag(), rho(1).real(),
               rho(1).imag(), rho(2).real(), rho(2).imag(), rho(3).real(),
               rho(3).imag(), paths.size());
    for (size_t i = 0; i < N; i++) {
        rho = extend_paths(paths, num_blips, i + 1);
        fmt::print(fp,
                   "{:^8f}\t{:+.15e}\t{:+.15e}\t{:+.15e}\t{:+.15e}\t{:+.15e}\t{"
                   ":+.15e}\t{:+.15e}\t{:+.15e}\t{:6d}\n",
                   ((i + 1) * dt), rho(0).real(), rho(0).imag(), rho(1).real(),
                   rho(1).imag(), rho(2).real(), rho(2).imag(), rho(3).real(),
                   rho(3).imag(), paths.size());
    }
    std::fclose(fp);
    return 0;
}
