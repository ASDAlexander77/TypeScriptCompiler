// A callback that assigns a captured string frees the string the variable held; a string borrowed
// across a call that may run it may be that string (spec 52). Not a borrow of anything else: the
// array walked is not a string, nor under one.
class Holder {
    constructor(public name: string, private data: string[] = []) {}

    each(f: (v: string) => void) {
        this.data.forEach(f);
    }
}

function label(i: number) {
    return "s" + i;
}

function main() {
    const h = new Holder(label(1), [label(2)]);
    let last = label(0);
    const name = h.name;
    h.each((v) => {
        last = v + name;
    });
    print(name, last);
}
