// `new` through a type alias that gives a generic class its type arguments keeps them:
// `type U8 = Buf<u8>; new U8([250])` is a Buf<u8>, where the constructor's arguments made it a
// Buf<s32>. And an array literal argument fits an overloaded constructor's `T[]` parameter as its
// elements fit T: `new Buf<u8>([250])` of a class with a second constructor found none for s32[].

class Buf<T> {
    data: T[];

    constructor(data: T[] = []) {
        this.data = data;
    }

    constructor(length: int) {
        this.data = [];
        for (let i = 0; i < length; i++) {
            this.data.push(<T>0);
        }
    }
}

type U8 = Buf<u8>;
type I16 = Buf<i16>;

class One<T> {
    constructor(public data: T[]) {}
}

type OneU8 = One<u8>;

function main() {
    const a = new Buf<u8>([250]);
    assert(sizeof<typeof a.data[0]>() == 1 && a.data[0] == 250, "an overloaded constructor takes [250] as u8[]");

    const b = new U8([250]);
    assert(sizeof<typeof b.data[0]>() == 1 && b.data[0] == 250, "the alias's u8");

    const c = new I16([-2, 3]);
    assert(sizeof<typeof c.data[0]>() == 2 && c.data[0] == -2, "the alias's i16");

    const d = new U8(3);
    assert(d.data.length == 3, "the other constructor");

    const e = new Buf<u8>([]);
    assert(e.data.length == 0, "an empty literal");

    const f = new OneU8([7]);
    assert(sizeof<typeof f.data[0]>() == 1, "one constructor");

    // without an alias, the arguments still give T
    const g = new One([1, 2]);
    assert(sizeof<typeof g.data[0]>() == 4, "inferred s32");

    print("done.");
}
