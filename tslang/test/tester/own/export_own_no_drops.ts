// -mm=own across a module boundary: the library side. `total` and `M.first` destroy nothing
// their caller can reach, and the library says so (`__tsown_<module>`); `shrink` removes an
// element of its argument, so it is not listed. See import_own_no_drops.ts.
export function total(a: number[]) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        s += a[i];
    }

    return s;
}

export namespace M {
    export function first(a: number[]) {
        return a.length > 0 ? a[0] : -1;
    }
}

export function shrink(a: number[][]) {
    a.pop();
}
