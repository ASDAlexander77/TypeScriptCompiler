import './export_c_varargs'

// See export_c_varargs.ts. printf returns how many characters it wrote: "1.5\n" is 4, a lost
// double prints "0\n", 2.
function main() {
    assert(cprintf("%g\n", 1.5) == 4);
    assert(cprintf("%d %g\n", 7, 12345.5) == 10);
    assert(callWith(cprintf) == 4);

    const p = new Printer(cprintf);
    assert(p.f("%g\n", 1.5) == 4);

    print("done.");
}
