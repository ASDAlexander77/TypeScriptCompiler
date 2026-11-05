// A number literal argument does not decide a generic's T that another argument already fixed:
// with `add<T>(ta: TA<T>, i: int, v: T)`, `add(new TA<u16>(...), 0, 2)` is T = u16 and 2 is cast to
// it, in either parameter order. Merging the two used to make T the literal's s32, and
// "type TA<ui16> is not assignable to type TA<si32>". A literal T can't hold still merges as before.

class TA<T> {
    constructor(public data: T[]) {}
}

function add<T>(ta: TA<T>, i: int, v: T): T {
    const old = ta.data[i];
    ta.data[i] = v;
    return old;
}

function addFirst<T>(v: T, ta: TA<T>, i: int): T {
    const old = ta.data[i];
    ta.data[i] = v;
    return old;
}

function same<T>(a: T, b: T): T {
    return b;
}

function main() {
    const t = new TA<u16>([65535]);
    assert(add(t, 0, 2) == 65535 && t.data[0] == 2, "the class's T first");
    assert(addFirst(3, t, 0) == 2 && t.data[0] == 3, "the literal first");
    assert(sizeof<typeof t.data[0]>() == 2, "u16");

    const s = new TA<i8>([-1]);
    assert(add(s, 0, -128) == -1 && s.data[0] == -128, "a negative literal an i8 holds");

    const u: u8 = 200;
    const r = same(u, 100);
    assert(r == 100 && sizeof<typeof r>() == 1, "T = u8");

    // 300 doesn't fit a u8: T is merged as before
    assert(same(u, 300) == 300, "a literal u8 can't hold");

    print("done.");
}
