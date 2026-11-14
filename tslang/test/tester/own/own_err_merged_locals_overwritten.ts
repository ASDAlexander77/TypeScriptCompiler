// TSLANG_ISSUES 58: a `const` declared from a merge of owning locals borrows each of them, so one of
// them given a new value while it is still used frees what the `const` may hold.
function mk(a: number): number[] { return [a, a]; }
function sum(v: number[]): number { return v[0] + v[1]; }

export function f(a: boolean): number {
    let t0 = mk(1.0);
    const t1 = mk(2.0);
    const s = a ? t0 : t1;
    t0 = mk(3.0);
    return sum(s);
}

function main() { print(f(true)); }
