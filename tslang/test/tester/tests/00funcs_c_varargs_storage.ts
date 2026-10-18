// A C variadic function held where a function value is stored - a class field, an object
// member, an optional, a `CVarArgs<...>` annotation - is still called the C variadic way. printf
// returns how many characters it wrote: "1.5\n" is 4, a lost double prints "0\n", 2.
@varargs
declare function printf(format: string): s32;

class Holder {
    f: typeof printf = printf;

    constructor(public g: typeof printf) {}
}

function pick(useIt: boolean): CVarArgs<(format: string) => s32> | undefined {
    return useIt ? printf : undefined;
}

function main() {
    const h = new Holder(printf);
    assert(h.f("%g\n", 1.5) == 4);
    assert(h.g("%g %d\n", 2.5, 7) == 6);

    const o = { f: printf };
    assert(o.f("%g\n", 1.5) == 4);

    const u = pick(true);
    if (u) {
        assert(u("%g\n", 1.5) == 4);
    }

    const c: CVarArgs<(format: string) => s32> = printf;
    assert(c("%g\n", 1.5) == 4);

    print("done.");
}
