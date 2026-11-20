// A class declared in a nested block of a function is that block's, as a `let` is: after the block the outer one
// is found again, and two blocks may declare the same name (each was the function's: "redefinition of symbol").
// A block at the module's top level is the same.
class C { x = 1; }

let fromBlocks: number = 0;
{
    class C { m = 10; }
    fromBlocks += new C().m;
}
{
    class C { n = 20; }
    fromBlocks += new C().n;
}

function make() {
    if (true) {
        class C { z = 3; get() { return this.z; } }
        return new C().get();
    }
    return 0;
}

function main() {
    {
        class C { y = 2; }
        const c = new C();
        assert(c.y == 2);
    }
    {
        class C { w = 4; }
        const c = new C();
        assert(c.w == 4);
    }
    for (let i = 0; i < 2; i++) {
        class D { v = 5; }
        assert(new D().v == 5);
    }
    const c = new C();
    assert(c.x == 1);
    assert(make() == 3);
    assert(fromBlocks == 30);
    print("done.");
}
