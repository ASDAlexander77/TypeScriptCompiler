// -mm=own, phase 4 rejects: `c` is the payload of the box `a` holds, borrowed through `___unbox`;
// assigning `a` destroys the box and its payload before `c.x` reads it.
class C {
    constructor(public x: number) {}
}

function main() {
    let a: any = new C(1);
    const c = <C>a;
    a = new C(2);
    print(c.x);
}
