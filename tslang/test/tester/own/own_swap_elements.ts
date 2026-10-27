// `[p, q] = [q, p]` of two elements or fields is an exchange (spec 42): each value keeps exactly one
// place, so nothing is retained, released or dropped, and under -mm=own an array of class instances
// can be reversed and sorted in place - through a local, `let v = a[i]`, the swap would make `v` a
// second owner. Reversed and sorted generically too, over strings built at run time. Each case churns
// between steps with blocks the size of the ones swapped, so a block freed by mistake is reused.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

class Pair {
    constructor(public a: C, public b: C) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

function made(n: int): C[] {
    const r: C[] = [];
    for (let i = 0; i < n; i++) {
        const c = new C(i);
        c.v.push(i);
        r.push(c);
    }

    return r;
}

function strings(n: int): string[] {
    const r: string[] = [];
    for (let i = 0; i < n; i++) r.push("s" + i);
    return r;
}

function reverseAll<T>(this: T[]) {
    const shift = this.length - 1;
    const size = this.length >> 1;
    for (let i = 0; i < size; i++) {
        [this[i], this[shift - i]] = [this[shift - i], this[i]];
    }

    return this;
}

function selectionSort<T>(a: T[], compare: (value1: T, value2: T) => int) {
    const len = a.length;
    for (let i = 0; i < len - 1; ++i) {
        for (let j = i + 1; j < len; ++j) {
            if (compare(a[i], a[j]) > 0) {
                [a[i], a[j]] = [a[j], a[i]];
            }
        }
    }
}

function byX(p: C, q: C): int {
    return p.x == q.x ? 0 : p.x > q.x ? 1 : -1;
}

function byText(p: string, q: string): int {
    return p == q ? 0 : p > q ? 1 : -1;
}

function inOrder(a: C[], first: number, step: number) {
    for (let i = 0; i < a.length; i++) {
        if (a[i].x != first + i * step || a[i].v[0] != a[i].x) return false;
    }

    return true;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 1000; i++) {
        // reversed in place, an odd and an even length; the result is the array it was given
        const a = made(5);
        const same = a.reverseAll();
        churn();
        if (inOrder(same, 4, -1)) ok++;

        const b = made(4);
        b.reverseAll();
        churn();
        if (inOrder(b, 3, -1)) ok++;

        // sorted with a comparison
        const c = made(6);
        c.reverseAll();
        [c[1], c[4]] = [c[4], c[1]];
        selectionSort(c, byX);
        churn();
        if (inOrder(c, 0, 1)) ok++;

        // a place swapped with itself keeps its value
        const k = 2;
        [c[k], c[2]] = [c[2], c[k]];
        churn();
        if (inOrder(c, 0, 1)) ok++;

        // two fields
        const p = new Pair(new C(1), new C(2));
        [p.a, p.b] = [p.b, p.a];
        churn();
        if (p.a.x == 2 && p.b.x == 1) ok++;

        // elements of two arrays, each of which then frees what it holds
        const d = made(2);
        const e = made(3);
        [d[0], e[2]] = [e[2], d[0]];
        churn();
        if (d[0].x == 2 && e[2].x == 0 && d[1].x == 1) ok++;

        // strings, generically
        const s = strings(5);
        s.reverseAll();
        churn();
        selectionSort(s, byText);
        churn();
        if (s[0] == "s0" && s[2] == "s2" && s[4] == "s4") ok++;
    }

    assert(ok == 1000 * 7, "ok");
    print("done.");
}
