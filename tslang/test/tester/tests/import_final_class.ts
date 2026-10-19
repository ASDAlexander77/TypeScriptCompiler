// A @final class from another module: its declaration (__decls) keeps @final, so the calls here are
// direct, and they link both against the import library and, under the JIT, to the DLL loaded at
// run time (where the class is a dynamic import and its calls stay vtable calls).
import './export_final_class'

function main() {
    const c = new Counter();
    for (let i = 0; i < 1000; i++) {
        c.inc();
    }

    assert(c.value == 1000);

    let sum = 0;
    for (const v of c.upTo(4)) {
        sum += v;
    }

    assert(sum == 10);
    print("done.");
}
