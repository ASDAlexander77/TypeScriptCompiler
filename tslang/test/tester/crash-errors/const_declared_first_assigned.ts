interface IT { v: number; m(): number }
function main() {
    const c: IT = { v: 5, m() { return c.v + 1; } };
    c = { v: 6, m() { return 1; } };
    print(c.m());
}