// A marker names a parameter the function does not have.
class C {
    constructor(public n: string) {}
}

export class Holder {
    c: C;

    @keeps("d")
    constructor(c: C) {
        this.c = c;
    }
}

function main() {
    print(new Holder(new C("a")).c.n);
}
