// -mm=own: two specializations of one generic class are not one class family. `Box<string>.find`
// returns a borrow of its field (through `pick`'s result) and `Box<int>.find` a number; neither
// class extends the other, so a call through either reaches only its own. Grouped by vtable slot and
// method name alone, they "disagreed" and both lost their facts: with Box<int> beside it,
// `Box<string>.find` was "'this value' borrows a field and cannot be stored, returned or captured".
function pick<T>(a: T[], f: (v: T) => boolean): T {
    for (const v of a) {
        if (f(v)) return v;
    }

    return a[0];
}

class Box<T> {
    constructor(private data: T[]) {}

    find(f: (v: T) => boolean) {
        return pick(this.data, f);
    }
}

function main() {
    const s = new Box<string>(["a", "bb"]);
    const n = new Box<int>([1, 2]);

    assert(s.find((v) => v.length == 2) == "bb");
    assert(n.find((v) => v > 1) == 2);

    print("done.");
}
