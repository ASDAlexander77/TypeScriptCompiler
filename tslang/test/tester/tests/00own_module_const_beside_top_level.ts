// TSLANG 191: a module `const` beside top-level code - a global the module's code stores once, where it is declared
// - read in a function across a call of a function value another such const holds, in every memory model. Under
// -mm=own it was refused ("a value that borrows 's' is used here after 's' is released or overwritten"): the call
// went through a global the module's code writes, so its callee was not known and it might write any global.
export const k = 5;
export const s = "x" + k;
export const o = { v: 7 };
export const g = function () { return k + 1; };
const inner = "in" + k;
const h = function (n: number) { return inner.length + n; };

export let label = "L";
export const relabel = function (n: number) { label = "L" + n; };

print(k, s, o.v, g(), h(1));

function readsAcross(): number {
    const t = s;
    const n = g() + h(2);
    return t.length + n + inner.length;
}

function main() {
    print(k, s, o.v, g());
    assert(s == "x5", "s");
    assert(readsAcross() == 2 + 6 + 5 + 3, "read across calls");

    // a call through such a const that writes a `let` of the module: what was read of the let before it is not
    // used after it, and the let is read again
    relabel(3);
    assert(label == "L3", "relabeled");
    const before = label;
    assert(before.length == 2, "label read");

    print("done.");
}
