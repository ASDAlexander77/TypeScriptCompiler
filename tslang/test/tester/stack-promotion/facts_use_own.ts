// Borrow facts under -mm=own: a program linked with --lib against facts_lib_own.

declare function sumOf(a: number[]): number;
declare function useOther(a: number[]): void;
declare namespace Geo {
    function len(a: number[]): number;
}

function otherKeep(a: number[]) {
}

function callSum(n: number) {
    const a = [n, n + 1, n + 2];
    return sumOf(a);
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
    callUseOther(1);
    print(callLen(1));
}
