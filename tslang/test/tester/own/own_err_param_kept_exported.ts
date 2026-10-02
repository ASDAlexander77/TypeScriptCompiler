// -mm=own rejects: the constructor keeps its parameter, which its callers would have to give up,
// but an exported class can be built from another module, which does not. (A string there is copied
// instead, spec 22.)
class C {
    constructor(public n: string) {}
}
export class Animal {
    c: C;
    constructor(c: C) {
        this.c = c;
    }
}
function main() {
    const a = new Animal(new C("cat"));
    print(a.c.n);
}
