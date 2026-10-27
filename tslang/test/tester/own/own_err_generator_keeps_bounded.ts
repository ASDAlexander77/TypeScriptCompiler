// -mm=own rejects: a generator that says `@keeps`, given one that borrows its argument, is bounded
// by that argument too, so it cannot be returned past it.
type It = { next(): { value: any, done: boolean } };

function* each(a: number[]) {
    for (const v of a) yield v;
}

@keeps("iter")
function* plusOne<TIter extends It>(iter: TIter) {
    for (const v of iter) yield v + 1;
}

function make() {
    const a: number[] = [];
    a.push(1);
    return plusOne(each(a));
}

function main() {
    for (const v of make()) print(v);
}
