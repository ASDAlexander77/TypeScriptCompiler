// -mm=own rejects: a generator method of a class another module may call keeps `this` in its state,
// and its callers keep owning it. It compiled, and the state's release freed the caller's object
// (the program then freed it again); `@borrows("this")` says what it does.
class S {
    constructor(public value: string) {}

    public *chars() {
        for (const c of this.value) yield <string>c;
    }
}

function main() {
    const s = new S("abc");
    for (const c of s.chars()) print(c);
}
