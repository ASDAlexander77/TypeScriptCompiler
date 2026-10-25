// __MEMORY_MODEL__ is the memory model the module is compiled for, a string literal, and a compare of
// it with a literal is decided while the module is generated: the branch not taken is not there at
// all. So an API only one model has can be called under a test of the model, and a callback that
// only gc needs kept - Boehm cannot see a pointer C holds - is not kept under own, whose caller does
// not own the object the callback holds.
type Callback = (n: number) => void;
let retained: Callback[] = [];

// defined nowhere: a call of it that were generated would fail to link (or to resolve, in the JIT)
declare function definedNowhere(): void;

class Registry {
    register(handler: Callback): void {
        if (__MEMORY_MODEL__ === "gc") {
            retained.push(handler);
        }

        handler(1);
    }
}

class Counter {
    count: number[] = [0.0];

    constructor(registry: Registry) {
        registry.register(this.onTick);
    }

    onTick(n: number): void {
        this.count[0] += n;
    }
}

function main() {
    if (__MEMORY_MODEL__ === "no such model") {
        definedNowhere();
    }

    const model: string = __MEMORY_MODEL__;
    assert(model == "gc" || model == "rc" || model == "own" || model == "none", "a memory model");
    assert(__MEMORY_MODEL__ !== "unknown", "a literal compare");

    // the target's OS and architecture, as Node names them
    const os: string = __OS__;
    const arch: string = __ARCH__;
    assert(os.length > 0 && arch.length > 0, "an OS and an architecture");
    if (__OS__ === "win32") {
        assert(os == "win32", "the branch for this OS");
    } else {
        assert(os != "win32", "the branch for another OS");
    }

    if (__ARCH__ === "x64" || __ARCH__ === "arm64") {
        assert(sizeof<Opaque>() == 8, "a 64-bit target");
    }

    const counter = new Counter(new Registry());
    assert(counter.count[0] == 1, "the callback ran");
    assert((retained.length == 1) == (model == "gc"), "kept under gc only");
    print(model, retained.length);
    print("done.");
}
