namespace Ifaces {
    interface IFoo {
        foo(): number;
    }
}

interface IFoo {
    foo(): number;
    foo2(): number;
}

class Cls1 implements Ifaces.IFoo, IFoo {
    foo(): number {
        print("Hello");
        return 1;
    }

    foo2(): number {
        print("Hello 2");
        return 2;
    }
}

function main() {
    const cls1 = new Cls1();
    assert(cls1.foo() == 1);

    const ifoo: Ifaces.IFoo = cls1;
    assert(ifoo.foo() == 1);

    const ifoo2: IFoo = cls1;
    assert(ifoo2.foo() == 1);
    // TODO:
    //assert(ifoo2.foo2() == 2);
    print("done.");
}
