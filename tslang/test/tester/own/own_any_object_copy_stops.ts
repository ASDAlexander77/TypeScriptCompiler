// -mm=own copies an `any` it cannot give a second owner (spec 71), but an object in it has one owner,
// the box: the copy stops the program with a message, as a failed assert does, after what was printed
// before it. Here one box is kept twice, which needs a copy for each keep.
class C {
    constructor(public n: number) {}
}

class Keeper {
    kept: any[] = [];

    keep(a: any) {
        this.kept.push(a);
    }
}

function main() {
    print("printed before");
    const k = new Keeper();
    const o: any = new C(1);
    k.keep(o);
    k.keep(o);
    print("not reached");
}
