// Borrow facts are keyed by the symbol a declaration binds: this `sumOf` binds facts_lib's `keep`,
// so facts_lib's `sumOf 0` must not apply to it.

@linkname("keep")
declare function sumOf(a: number[]): void;
declare function keptLength(): number;

function otherKeep(a: number[]) {
}

function call(n: number) {
    const a = [n, n, n];
    sumOf(a);
}

function main() {
    call(1);
    print(keptLength());
}
