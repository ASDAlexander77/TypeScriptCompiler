// -mm=own rejects: a generator that borrows its argument cannot be given to one whose state keeps
// what it is given - it would outlive what it borrows.
type It = { next(): { value: any, done: boolean } };

function* each(a: number[]) {
    for (const v of a) yield v;
}

@keeps("iter")
function* plusOne<TIter extends It>(iter: TIter) {
    for (const v of iter) yield v + 1;
}

function main() {
    const a = [1, 2, 3];
    for (const v of plusOne(each(a))) print(v);
}
