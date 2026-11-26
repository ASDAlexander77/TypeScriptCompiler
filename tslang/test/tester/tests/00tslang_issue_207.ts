// TSLANG 207 (leftovers of 195, 196 and 205), every model.

// 205: a literal given to a generic call whose function member returns an integer, where the type parameter is a
// number from another argument: the member returns the number its receiver says ("can't assign '() => s32' to
// '() => number'"), in either order of the arguments
interface Options<Data> {
    produce: () => Data;
}

function takeG<D>(o: Options<D>, d: D): D {
    return o.produce();
}

function takeG2<D>(d: D, o: Options<D>): D {
    return o.produce();
}

// 196: a literal that leaves out an optional field of a tuple it is given to, holding a function (under -mm=own "a
// value that borrows a field cannot be stored")
type Cb = () => number;

function main() {
    assert(takeG({ produce: () => 44 }, 1.5) / 8 == 5.5);
    assert(takeG2(1.5, { produce: () => 44 }) / 8 == 5.5);
    assert(takeG({ produce: () => 4 }, 1) == 4);

    const cb: Cb = () => 3;
    const o: { disc?: false; cb: Cb } = { cb };
    assert(o.cb() == 3 && o.disc === undefined);

    const p: { n?: number; cb: Cb; m?: string } = { cb: cb };
    assert(p.cb() == 3 && p.n === undefined && p.m === undefined);

    const q: { n?: number; s: string } = { s: "s" + 1 };
    assert(q.s == "s1" && q.n === undefined);

    print("done.");
}
