// TSLANG_ISSUES 58 (-mm=own), from native_ts_graphics' core/directx_math.ts (XMQuaternionRotationMatrix):
// `const selected = c1 ? (c2 ? tensor0 : tensor1) : (c3 ? tensor2 : tensor3)` over four owned locals,
// then only read. Each local keeps its value, so `selected` borrows whichever one it was given - in a
// loop's body too, and with null as one way in.

function mk(a: number): number[] {
    return [a, a + 1.0, a + 2.0, a + 3.0];
}

function sum(v: number[]): number {
    let s = 0.0;
    for (let i = 0; i < v.length; i++) {
        s += v[i];
    }

    return s;
}

function scaled(v: number[], s: number): number[] {
    return [v[0] / s, v[1] / s, v[2] / s, v[3] / s];
}

export function pick(a: boolean, b: boolean, c: boolean): number[] {
    const tensor0 = mk(1.0);
    const tensor1 = mk(2.0);
    const tensor2 = mk(3.0);
    const tensor3 = mk(4.0);
    const selected = a ? (b ? tensor0 : tensor1) : (c ? tensor2 : tensor3);
    return scaled(selected, sum(selected));
}

export function total(a: boolean): number {
    let r = 0.0;
    for (let i = 0; i < 3; i++) {
        const t0 = mk(i);
        const t1 = mk(10.0);
        const s = a ? t0 : t1;
        r += sum(s);
    }

    return r;
}

export function outer(a: boolean): number {
    const t0 = mk(1.0);
    const t1 = mk(2.0);
    let r = 0.0;
    for (let i = 0; i < 2; i++) {
        const s = i == 0 ? t0 : t1;
        r += sum(s);
    }

    const n = a ? t0 : null;
    return n ? r + sum(n) : r;
}

export function freshInLoop(a: boolean): number {
    let r = 0.0;
    for (let i = 0; i < 3; i++) {
        const p = a ? mk(i) : null;
        if (p) {
            r += sum(p);
        }
    }

    return r;
}

function main() {
    const p = pick(false, true, false);
    print(p[0], p[3]);
    assert(p[0] == 4.0 / 22.0 && p[3] == 7.0 / 22.0);
    assert(pick(true, true, true)[0] == 0.1 && pick(true, false, true)[0] == 2.0 / 14.0);
    assert(total(true) == 6.0 + 10.0 + 14.0 && total(false) == 3.0 * 46.0);
    assert(outer(true) == 10.0 + 14.0 + 10.0 && outer(false) == 24.0);
    assert(freshInLoop(true) == 30.0 && freshInLoop(false) == 0.0);
    print("done.");
}
