// TSLANG 190: a value wider than a pointer - a function value's {func, this, tag} - read out of an `any` whose
// box holds a smaller one (`null`) read up to 16 bytes past the end of the box; an unoptimized build kept the
// whole load, and a guard-page heap faulted on it (00any_call under -mm=rc, --opt=false). Only the bytes the
// box has are read now: the rest is zero, a null function.
function twice(n: number) {
    return n * 2;
}

function apply(f: ((n: number) => number) | null, n: number) {
    return f ? f(n) : -1;
}

function main() {
    for (let i = 0; i < 1000; i++) {
        const nothing: any = null;
        const maybe = <((n: number) => number) | null>nothing;
        assert(maybe === null, "null to a function or null");
        assert(apply(maybe, i) == -1, "null given on");

        const captured = i;
        const add: any = (n: number) => n + captured;
        const back = <((n: number) => number) | null>add;
        assert(back !== null && back(1) == i + 1, "a closure read back whole");

        const plain: any = twice;
        assert(apply(<(n: number) => number>plain, i) == i * 2, "a function read back whole");
    }

    print("done.");
}
