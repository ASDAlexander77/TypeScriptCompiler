// TSLANG_ISSUES 29: a function value - a `const` of function type, a callback parameter - given to a
// @cdecl function's callback parameter. A plain function passes as a C function pointer; a closure,
// whose captures the pointer can't carry, throws there instead of running later with none.
@cdecl declare function atexit(fn: () => void): s32;

function plain() {
    print("plain at exit");
}

function register(f: () => void) {
    atexit(f);
}

function main() {
    let n = 42;
    const viaPlain: () => void = plain;
    atexit(viaPlain);
    register(plain);

    let caught = false;
    try {
        const viaClosure: () => void = () => {
            print("closure sees", n);
        };
        register(viaClosure);
    } catch (e) {
        caught = true;
    }

    assert(caught, "a closure as a C function pointer throws");
    print("done.");
}
