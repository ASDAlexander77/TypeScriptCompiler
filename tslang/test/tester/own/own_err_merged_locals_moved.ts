// TSLANG_ISSUES 58: a `const` declared from a merge of owning locals borrows each of them, so none of
// them can be moved out while it does.
function mk(a: number): number[] { return [a, a]; }
function sum(v: number[]): number { return v[0] + v[1]; }

let kept: number[] = [];

export function f(a: boolean): number {
    const t0 = mk(1.0);
    const t1 = mk(2.0);
    const s = a ? t0 : t1;
    kept = t0;
    return sum(s);
}

function main() { print(f(true)); }
