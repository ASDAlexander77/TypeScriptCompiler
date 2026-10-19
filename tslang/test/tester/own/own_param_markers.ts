// -mm=own: what a function's signature says - the parameters it keeps (@keeps), the arguments its
// result may not outlive (@borrows) - holds for an exported function too, whose callers this module
// does not all see. The other memory models ignore both.
class Tag {
    constructor(public name: string) {}
}

export class Holder {
    tag: Tag;

    @keeps("tag")
    constructor(tag: Tag) {
        this.tag = tag;
    }
}

export class Word {
    constructor(public text: string) {}

    // a generator method over `this`, which its state borrows
    @borrows("this")
    *letters() {
        for (const c of this.text) yield <string>c;
    }
}

// a generator whose state borrows its argument: the caller owns it, but it may not outlive `w`
@borrows("w")
export function *lettersOf(w: Word) {
    for (const c of w.text) yield <string>c;
}

// returns what a call with a bounded result gives it, so it is bound to the same argument
@borrows("w")
export function forward(w: Word) {
    return lettersOf(w);
}

function main() {
    const h = new Holder(new Tag("kept"));
    assert(h.tag.name == "kept");

    const w = new Word("abc");
    let s = "";
    for (const c of lettersOf(w)) s += c;
    for (const c of forward(w)) s += c;
    for (const c of w.letters()) s += c;
    assert(s == "abcabcabc");

    print("done.");
}
