// Borrow facts: a program linked with --lib against facts_lib. A fresh array goes on the stack
// when the library listed the parameter it is passed to, and stays on the heap otherwise. Arrays
// are built from a parameter: a constant literal is a static global, never allocated.

declare function sumOf(a: number[]): number;
declare function keep(a: number[]): void;
declare function same(a: number[]): number[];
declare function viaCallback(a: number[], f: (x: number[]) => void): void;
declare function mixed(a: number[], b: number[]): number;
declare function useOther(a: number[]): void;
declare function keptLength(): number;
declare namespace Geo {
    function len(a: number[]): number;
}

function otherKeep(a: number[]) {
}

function callSum(n: number) {
    const a = [n, n + 1, n + 2];
    return sumOf(a);
}

function callKeep(n: number) {
    const a = [n, n];
    keep(a);
}

function callSame(n: number) {
    const a = [n];
    return same(a).length;
}

function callViaCallback(n: number) {
    const a = [n, n];
    let seen = 0;
    viaCallback(a, (x: number[]) => { seen = x.length; });
    return seen;
}

function callMixed(n: number) {
    const a = [n];
    const b = [n, n];
    return mixed(a, b);
}

function callUseOther(n: number) {
    const a = [n];
    useOther(a);
}

function callLen(n: number) {
    const a = [n, n, n, n];
    return Geo.len(a);
}

function main() {
    print(callSum(1));
    callKeep(1);
    print(keptLength());
    print(callSame(1));
    print(callViaCallback(1));
    print(callMixed(1));
    print(keptLength());
    callUseOther(1);
    print(callLen(1));
}
