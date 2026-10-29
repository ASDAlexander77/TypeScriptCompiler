// A class whose constructor takes other arguments than its base's is no value of `typeof Base`:
// `new k(1)` would give it a number for its string.
class Base {
    constructor(public n: number) {}
}

class Named extends Base {
    constructor(public s: string) {
        super(0);
    }
}

function main() {
    const k: typeof Base = Named;
    print(new k(1).n);
}
