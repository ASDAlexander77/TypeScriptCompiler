// Borrow facts under -mm=own, which refuses a function that keeps or returns its parameter: what
// is left is a parameter only read, one handed to a function with no body, and a virtual call.

declare function otherKeep(a: number[]): void;

export function sumOf(a: number[]): number {
    let s = 0;
    for (const v of a) s += v;
    return s;
}

export function useOther(a: number[]) {
    otherKeep(a);
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
