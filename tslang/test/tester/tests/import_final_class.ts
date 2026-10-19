// A @final class from another module, imported from its DLL (test-runner -shared builds the DLL
// first, so the import is dynamic): its declaration (__decls) keeps @final, and its calls stay
// vtable calls, as every call into a class declared here does. The static import is
// import_final_base_static.ts.
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
