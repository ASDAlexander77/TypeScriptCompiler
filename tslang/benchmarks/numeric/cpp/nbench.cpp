// C++ port of tslang/benchmarks/numeric/nbench.ts: the same six kernels over std::vector<double> /
// std::vector<int>, printing the same six check values. Build with -ffp-contract=off: clang
// otherwise fuses a*b+c into one instruction where the CPU has it, which changes the low digits.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static const int N = 4096;
static const int M = 64;

static double check(double v) { return std::fmod(std::floor(v * 1000.0), 999983.0); }

static std::vector<double> makeArray(int n, double mul, int mod, double add) {
    std::vector<double> a(n);
    for (int i = 0; i < n; i++) a[i] = (i % mod) * mul + add;
    return a;
}

static double sum(const std::vector<double> &a) {
    double s = 0;
    for (double v : a) s += v;
    return s;
}

__attribute__((noinline)) static void saxpy(double a, const std::vector<double> &x, std::vector<double> &y, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

__attribute__((noinline)) static void triad(const std::vector<double> &x, const std::vector<double> &y,
                                            std::vector<double> &z, double c, int n) {
    for (int i = 0; i < n; i++) z[i] = x[i] * y[i] + c;
}

__attribute__((noinline)) static void stencil(const std::vector<double> &src, std::vector<double> &dst, int n) {
    for (int i = 1; i < n - 1; i++) dst[i] = (src[i - 1] + 2.0 * src[i] + src[i + 1]) * 0.25;
}

__attribute__((noinline)) static double dot(const std::vector<double> &x, const std::vector<double> &y, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}

__attribute__((noinline)) static void matmul(const std::vector<double> &a, const std::vector<double> &b,
                                             std::vector<double> &c, int m) {
    for (int i = 0; i < m; i++)
        for (int k = 0; k < m; k++) {
            const double aik = a[i * m + k];
            for (int j = 0; j < m; j++) c[i * m + j] += aik * b[k * m + j];
        }
}

__attribute__((noinline)) static int hash(std::vector<int> &h, const std::vector<int> &g, int n) {
    uint32_t s = 0;
    for (int i = 0; i < n; i++) {
        h[i] = (int)(((uint32_t)h[i] * 31u + (uint32_t)g[i]) & 0xffffu);
        s += (uint32_t)h[i];
    }
    return (int)s;
}

static double runSaxpy() {
    auto x = makeArray(N, 0.5, 97, 1.0);
    auto y = makeArray(N, 0.25, 89, 2.0);
    for (int r = 0; r < 60000; r++) saxpy(0.000001, x, y, N);
    return check(sum(y));
}

static double runTriad() {
    auto x = makeArray(N, 0.5, 97, 1.0);
    auto y = makeArray(N, 0.25, 89, 2.0);
    auto z = makeArray(N, 0.0, 1, 0.0);
    double s = 0;
    for (int r = 0; r < 60000; r++) {
        triad(x, y, z, r * 0.001, N);
        s += z[r % N];
    }
    return check(s + sum(z));
}

static double runStencil() {
    auto a = makeArray(N, 0.5, 97, 1.0);
    auto b = makeArray(N, 0.5, 97, 1.0);
    for (int r = 0; r < 30000; r++) {
        stencil(a, b, N);
        stencil(b, a, N);
    }
    return check(sum(a));
}

static double runDot() {
    auto x = makeArray(N, 0.5, 97, 1.0);
    auto y = makeArray(N, 0.25, 89, 2.0);
    double s = 0;
    for (int r = 0; r < 30000; r++) {
        x[r % N] += 0.5;
        s += dot(x, y, N) * 0.000001;
    }
    return check(s);
}

static double runMatmul() {
    auto a = makeArray(M * M, 0.001, 13, 0.01);
    auto b = makeArray(M * M, 0.002, 7, 0.02);
    auto c = makeArray(M * M, 0.0, 1, 0.0);
    for (int r = 0; r < 400; r++) matmul(a, b, c, M);
    return check(sum(c));
}

static int runHash() {
    std::vector<int> h(N), g(N);
    for (int i = 0; i < N; i++) {
        h[i] = i % 251;
        g[i] = i % 241 + 7;
    }
    uint32_t s = 0;
    for (int r = 0; r < 60000; r++) s = (s + (uint32_t)hash(h, g, N)) & 0xfffffu;
    return (int)s;
}

// With no argument every kernel runs; with a kernel's name only that one does.
int main(int argc, char **argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    auto run = [&](const char *name) { return only.empty() || only == name; };
    std::printf("start...\n");
    if (run("saxpy")) std::printf("saxpy: %.0f\n", runSaxpy());
    if (run("triad")) std::printf("triad: %.0f\n", runTriad());
    if (run("stencil")) std::printf("stencil: %.0f\n", runStencil());
    if (run("dot")) std::printf("dot: %.0f\n", runDot());
    if (run("matmul")) std::printf("matmul: %.0f\n", runMatmul());
    if (run("hash")) std::printf("hash: %d\n", runHash());
    std::printf("done.\n");
}
