// Root-level executable code coexisting with ordinary function/class
// declarations, still with no explicit `function main()` anywhere - only the
// root-level STATEMENTS (not the helper declarations) become the synthesized
// `main` body (generateGlobalEntryCode in MLIRGenModule.cpp); the helpers stay
// ordinary top-level declarations, callable from that synthesized body exactly
// like they'd be callable from a hand-written main().

function double(x: number): number {
    return x * 2;
}

class Box {
    value: number;
    constructor(v: number) {
        this.value = v;
    }
    get(): number {
        return this.value;
    }
}

const results: number[] = [];
for (let i = 1; i <= 3; i++) {
    results.push(double(i));
}
assert(results[0] == 2 && results[1] == 4 && results[2] == 6, "helpers-from-root");

const box = new Box(21);
assert(double(box.get()) == 42, "class-and-function-mixed");

print("done.");
