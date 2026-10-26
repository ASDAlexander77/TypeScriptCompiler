// -mm=own rejects: the generator `w.letters()` borrows `w`, through the vtable of a class that
// is not `@final`, and `w` is given a new value while the generator is still read. It compiled when
// the method kept `this`; then the bound check did not see the object through `ts.GetThis`.
class Word {
    constructor(public text: string) {}

    @borrows("this")
    *letters() {
        for (const c of this.text) yield <string>c;
    }
}

function main() {
    let w = new Word("abc");
    const it = w.letters();
    w = new Word("zz");
    for (const c of it) print(c);
}
