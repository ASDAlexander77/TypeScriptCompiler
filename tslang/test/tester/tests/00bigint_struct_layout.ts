// regression test: a `bigint` (an i64) placed after a smaller field inside a non-packed struct.
//
// Lowering used to build its type converter with LLVM's default data layout, in which i64 is
// ABI-aligned at 4 bytes, instead of the target's (x86-64 aligns i64 at 8; i686-msvc too). The
// only values whose layout that difference can move are these: an optional bigint ({i64, i1}), a
// class/object/tuple with a bigint after a boolean ({i1, i64}), and arrays of them. None of the
// rest of the corpus uses bigint, so this file is what exercises it. Every value is written and
// read back, through fields, elements and optionals, and checked by comparison.
//
// The union at the end is the case that was actually miscompiled on x64. A union's storage is its
// largest member, chosen with the converter's layout: {i1, i64} measured 12 bytes there, the same
// as {i1, i32, i32}, so the tie went to the s32 member. Storing the bigint member then copied only
// 20 of its 24 bytes (tag + 12-byte storage) and dropped the top half of the i64 -- the read-back
// below saw 1 instead of 9007199254740993. With the real layout {i1, i64} is 16 bytes and wins.
//
// Printing a bigint is deliberately avoided: bigint -> string is not implemented (it lowers to an
// inttoptr of the value), so every check here is a comparison against a bigint literal.

type Three = { flag: boolean; a: s32; b: s32 };
type Big = { flag: boolean; v: bigint };

function pickUnion(three: boolean): Three | Big {
    if (three) {
        const t: Three = { flag: false, a: 2, b: 3 };
        return t;
    }
    const b: Big = { flag: true, v: 9007199254740993n };
    return b;
}

class Pair {
    a: boolean;
    b: bigint;
    constructor(a: boolean, b: bigint) {
        this.a = a;
        this.b = b;
    }
}

function optionalValue(v: bigint | undefined): bigint {
    if (v === undefined) return -1n;
    return v;
}

function main() {
    // optional bigint: {i64, i1}
    let x: bigint | undefined = 1n;
    assert(x == 1n, "optional x");
    if (x !== undefined) {
        assert(x + 1n == 2n, "optional x + 1");
    }
    let y: bigint | undefined = undefined;
    assert(y === undefined, "optional y");
    assert(optionalValue(9007199254740993n) == 9007199254740993n, "optional arg");
    assert(optionalValue(undefined) == -1n, "optional arg undefined");

    // class: a boolean, then a bigint
    const c = new Pair(true, 9007199254740993n);
    assert(c.a, "class a");
    assert(c.b == 9007199254740993n, "class b");
    c.b = c.b + 1n;
    c.a = false;
    assert(!c.a, "class a written");
    assert(c.b == 9007199254740994n, "class b written");

    // tuple: [boolean, bigint]
    let t: [boolean, bigint] = [true, 42n];
    assert(t[0], "tuple 0");
    assert(t[1] == 42n, "tuple 1");
    t[1] = -5000000000n;
    assert(t[1] == -5000000000n, "tuple 1 written");
    const [p, q] = t;
    assert(p, "tuple destructured 0");
    assert(q == -5000000000n, "tuple destructured 1");

    // object literal: { a: boolean, b: bigint }
    const o = { a: false, b: 77n };
    assert(!o.a, "object a");
    assert(o.b == 77n, "object b");

    // arrays: the element stride is the struct's size, 16 bytes on x64 either way
    const pairs: [boolean, bigint][] = [[true, 1n], [false, 4294967297n], [true, -3n]];
    assert(pairs[1][1] == 4294967297n, "array element 1");
    assert(pairs[2][1] == -3n, "array element 2");
    assert(!pairs[1][0], "array element 1 flag");
    let sum = 0n;
    for (const e of pairs) sum = sum + e[1];
    assert(sum == 4294967295n, "array sum");

    const opts: (bigint | undefined)[] = [5n, undefined, 6n];
    assert(opts[0] == 5n, "optional array 0");
    assert(opts[1] === undefined, "optional array 1");
    assert(opts[2] == 6n, "optional array 2");

    const objs = [new Pair(true, 10n), new Pair(false, 20n)];
    assert(objs[1].b == 20n, "class array");

    // union whose members are {i1, i32, i32} and {i1, i64}: the storage must hold the bigint whole
    const big = <Big><any>pickUnion(false);
    assert(big.flag, "union big flag");
    assert(big.v == 9007199254740993n, "union big value");
    // The other member must read back whole too. `a` sits at offset 4, inside the padding of the
    // {i1, i64} member; union storage has to carry those bytes (see 00union_member_padding).
    const three = <Three><any>pickUnion(true);
    assert(!three.flag, "union three flag");
    assert(three.a == 2, "union three a");
    assert(three.b == 3, "union three b");

    print("done.");
}
