class SuperCls1
{
    public s: int;
}

class Cls1 extends SuperCls1
{
    public i: int;
}

function main() {

    const a = new Cls1();
    a.i = 10;
    a.s = 20;

    const box = a as any;

    const unboxed = box as Cls1;

    assert(unboxed.s == 20);

    print("done.");
}
