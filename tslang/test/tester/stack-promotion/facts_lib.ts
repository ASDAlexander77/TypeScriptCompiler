// Borrow facts (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md): what
// this library's __tsborrow_ symbol lists - the parameters its own analysis shows are only read.

declare function otherKeep(a: number[]): void;

let kept: number[] = [];

export function sumOf(a: number[]): number {
    let s = 0;
    for (const v of a) s += v;
    return s;
}

export function keep(a: number[]) {
    kept = a;
}

export function same(a: number[]): number[] {
    return a;
}

export function viaCallback(a: number[], f: (x: number[]) => void) {
    f(a);
}

export function mixed(a: number[], b: number[]): number {
    kept = b;
    return a[0];
}

export function useOther(a: number[]) {
    otherKeep(a);
}

function helper(a: number[]): number {
    return a.length;
}

export function callsHelper(): number {
    return helper([1, 2]);
}

export namespace Geo {
    export function len(a: number[]): number {
        return a.length;
    }
}

export class Holder {
    peek(a: number[]): number {
        return a[0] + this.twice(a);
    }

    twice(a: number[]): number {
        return a[0] * 2;
    }
}

export function makeHolder(): Holder {
    return new Holder();
}

export function keptLength(): number {
    return kept.length;
}

// defines `renamedLen`, not `lenOf`: no fact may name either
@linkname("renamedLen")
export function lenOf(a: number[]): number {
    return a.length;
}
