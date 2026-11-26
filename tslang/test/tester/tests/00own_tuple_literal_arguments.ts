// A tuple literal given to a call (`h(["a" + i, 1])`), or in an array literal given to one (`g([["a" + i, 1]])`):
// under -mm=own the new string went into the tuple, which nothing gave back - each call leaked the string
// (and the array of tuples when it went to an optional parameter, as `new Map([["a" + i, 1]])` does).
// Runtime strings, not literals: a literal is immortal and hides a leak or a wrong count.
function g(a: [string, number][]): number {
    let n: number = 0;
    for (const t of a) {
        n += t[0].length + t[1];
    }

    return n;
}

function h(t: [string, number]): number {
    return t[0].length + t[1];
}

function opt(a?: [string, number][]): number {
    return a ? g(a) : -1;
}

function fromParam(s: string): number {
    // the tuple borrows a string its caller owns: nothing may be given back here
    return g([[s, 1]]) + h([s, 2]);
}

function main() {
    assert(h(["ab", 1]) == 3, "constant tuple");
    assert(g([["ab", 1], ["c", 2]]) == 6, "constant tuples");
    assert(opt([["abc", 1]]) == 4, "constant tuples to an optional parameter");

    let n: number = 0;
    for (let i = 0; i < 20000; i++) {
        const s = "k" + i;
        n += h(["a" + i, 1]);
        n += g([["a" + i, 1], ["b" + i, 2]]);
        n += opt([["c" + i, 3]]);
        n += opt([["d", 4]]);
        n += fromParam(s);
        n += s.length > 0 ? 1 : 0;
    }

    assert(n > 0, "many times");
    print(h(["x" + n, 0]));

    print("done.");
}
