// -mm=own rejects: a generator that says `@keeps`, given one that borrows its argument, used after
// that argument is overwritten.
type It = { next(): { value: any, done: boolean } };

function* each(a: number[]) {
    for (const v of a) yield v;
}

@keeps("iter")
function* plusOne<TIter extends It>(iter: TIter) {
    for (const v of iter) yield v + 1;
}

function main() {
    let a: number[] = [];
    a.push(1, 2);
    const g = plusOne(plusOne(each(a)));
    print(g.next().value);
    a = [];
    print(g.next().value);
}
