#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

constexpr int kGrid = 128;
constexpr int kPitch = kGrid + 2;
constexpr double kDt = 1.0e-2;
constexpr double kR = 8.314;
constexpr double kMobility = 1.0;
constexpr double kGradient = 3.57e-1;
constexpr double kMeanComposition = 0.60;
constexpr double kTemperature = 973.15;

struct Options {
    std::uint64_t seed = 12345;
    int steps = 52000;
    int save_start = 2000;
    int save_interval = 10;
    int threads = 1;
    std::string input_path;
};

int index_of(int row, int column) { return column + row * kPitch; }

int parse_int(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > 100000000L) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

std::uint64_t parse_seed(const char* value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid seed: ") + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int position = 1; position < argc; ++position) {
        const std::string argument = argv[position];
        if (position + 1 >= argc) {
            throw std::runtime_error("missing value for " + argument);
        }
        const char* value = argv[++position];
        if (argument == "--seed") {
            options.seed = parse_seed(value);
        } else if (argument == "--steps") {
            options.steps = parse_int(value, "steps");
        } else if (argument == "--save-start") {
            options.save_start = parse_int(value, "save-start");
        } else if (argument == "--save-interval") {
            options.save_interval = parse_int(value, "save-interval");
        } else if (argument == "--threads") {
            options.threads = parse_int(value, "threads");
        } else if (argument == "--input") {
            options.input_path = value;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.save_interval <= 0 || options.threads <= 0) {
        throw std::runtime_error("save-interval and threads must be positive");
    }
    if (options.save_start > options.steps) {
        throw std::runtime_error("save-start must not exceed steps");
    }
    if ((options.steps - options.save_start) % options.save_interval != 0) {
        throw std::runtime_error("steps-save-start must be divisible by save-interval");
    }
    return options;
}

void apply_periodic_boundary(std::vector<double>& field) {
    for (int column = 1; column <= kGrid; ++column) {
        field[index_of(0, column)] = field[index_of(kGrid, column)];
        field[index_of(kGrid + 1, column)] = field[index_of(1, column)];
    }
    for (int row = 1; row <= kGrid; ++row) {
        field[index_of(row, 0)] = field[index_of(row, kGrid)];
        field[index_of(row, kGrid + 1)] = field[index_of(row, 1)];
    }
    field[index_of(0, 0)] = field[index_of(kGrid, kGrid)];
    field[index_of(0, kGrid + 1)] = field[index_of(kGrid, 1)];
    field[index_of(kGrid + 1, 0)] = field[index_of(1, kGrid)];
    field[index_of(kGrid + 1, kGrid + 1)] = field[index_of(1, 1)];
}

void initialize(std::vector<double>& concentration, const Options& options) {
    if (!options.input_path.empty()) {
        std::ifstream input(options.input_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open input: " + options.input_path);
        }
        std::vector<double> interior(static_cast<std::size_t>(kGrid) * kGrid);
        input.read(reinterpret_cast<char*>(interior.data()),
                   static_cast<std::streamsize>(interior.size() * sizeof(double)));
        if (input.gcount() != static_cast<std::streamsize>(interior.size() * sizeof(double))) {
            throw std::runtime_error("input does not contain one 128x128 float64 field");
        }
        char extra = 0;
        if (input.read(&extra, 1)) {
            throw std::runtime_error("input contains trailing bytes");
        }
        for (int row = 1; row <= kGrid; ++row) {
            for (int column = 1; column <= kGrid; ++column) {
                concentration[index_of(row, column)] =
                    interior[static_cast<std::size_t>(row - 1) * kGrid + (column - 1)];
            }
        }
        return;
    }

    std::mt19937_64 generator(options.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int row = 1; row <= kGrid; ++row) {
        for (int column = 1; column <= kGrid; ++column) {
            const double sample = uniform(generator);
            concentration[index_of(row, column)] =
                kMeanComposition + 0.05 * (0.5 - sample);
        }
    }
}

void calculate_chemical_potential(const std::vector<double>& concentration,
                                  std::vector<double>& chemical_potential) {
    const double a0 = 15000.0 + 6.1 * kTemperature;
    const double a1 = -7600.0 + 3.55 * kTemperature;
    const double rt = kR * kTemperature;
#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 1; row <= kGrid; ++row) {
        for (int column = 1; column <= kGrid; ++column) {
            const int center = index_of(row, column);
            double value = concentration[center];
            if (value <= 0.0) value = 1.0e-10;
            if (value >= 1.0) value = 1.0 - 1.0e-10;
            const double derivative =
                (rt * std::log(value / (1.0 - value)) +
                 (1.0 - 2.0 * value) * a0 +
                 (-6.0 * value + 6.0 * value * value + 1.0) * a1) /
                rt;
            const double laplacian =
                concentration[index_of(row + 1, column)] +
                concentration[index_of(row - 1, column)] +
                concentration[index_of(row, column + 1)] +
                concentration[index_of(row, column - 1)] -
                4.0 * concentration[center];
            chemical_potential[center] = derivative - 2.0 * kGradient * laplacian;
        }
    }
}

void update_concentration(std::vector<double>& concentration,
                          const std::vector<double>& chemical_potential) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int row = 1; row <= kGrid; ++row) {
        for (int column = 1; column <= kGrid; ++column) {
            const int center = index_of(row, column);
            const double laplacian =
                chemical_potential[index_of(row + 1, column)] +
                chemical_potential[index_of(row - 1, column)] +
                chemical_potential[index_of(row, column + 1)] +
                chemical_potential[index_of(row, column - 1)] -
                4.0 * chemical_potential[center];
            double value = concentration[center] + kDt * kMobility * laplacian;
            value = std::max(0.0, std::min(1.0, value));
            concentration[center] = value;
        }
    }
}

void write_frame(const std::vector<double>& concentration, std::vector<float>& frame) {
    for (int row = 1; row <= kGrid; ++row) {
        for (int column = 1; column <= kGrid; ++column) {
            const double serialized =
                std::round(concentration[index_of(row, column)] * 1.0e6) / 1.0e6;
            frame[static_cast<std::size_t>(row - 1) * kGrid + (column - 1)] =
                static_cast<float>(serialized);
        }
    }
    const std::size_t written = std::fwrite(frame.data(), sizeof(float), frame.size(), stdout);
    if (written != frame.size()) {
        throw std::runtime_error("failed to stream a complete output frame");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads);
        std::setvbuf(stdout, nullptr, _IOFBF, 1U << 20U);

        std::vector<double> concentration(static_cast<std::size_t>(kPitch) * kPitch, 0.0);
        std::vector<double> chemical_potential(static_cast<std::size_t>(kPitch) * kPitch, 0.0);
        std::vector<float> frame(static_cast<std::size_t>(kGrid) * kGrid);
        initialize(concentration, options);

        int frames = 0;
        if (options.save_start == 0) {
            write_frame(concentration, frame);
            ++frames;
        }
        for (int step = 1; step <= options.steps; ++step) {
            apply_periodic_boundary(concentration);
            calculate_chemical_potential(concentration, chemical_potential);
            apply_periodic_boundary(chemical_potential);
            update_concentration(concentration, chemical_potential);
            if (step >= options.save_start &&
                (step - options.save_start) % options.save_interval == 0) {
                write_frame(concentration, frame);
                ++frames;
            }
            if (step % 5000 == 0) {
                std::cerr << "step=" << step << "/" << options.steps << " frames=" << frames << '\n';
            }
        }
        std::fflush(stdout);
        std::cerr << "complete steps=" << options.steps << " frames=" << frames
                  << " threads=" << options.threads << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "spinodal_solver error: " << error.what() << '\n';
        return 1;
    }
}
