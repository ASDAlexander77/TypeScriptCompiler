class SuperCls1
{
    public s: int;
}

class Cls1 extends SuperCls1
{
    public i: int;
}

interface IFce1
{
    i: int;
}

function class_any()
{
    const a = new Cls1();
    a.i = 10;
    a.s = 20;

    const box = a as any;

    const unboxed = box as Cls1;

    assert(unboxed.s == 20);
}

function array_any()
{
    let a = [1, 2, 3];

    const box = a as any;

    const unboxed = box as int[];    

    assert(unboxed.length == a.length);
}

function iface_any()
{
    const c = new Cls1();
    c.i = 11;

    const iface = c as IFce1;

    const box = iface as any;

    const unboxed = box as IFce1;

    assert(unboxed.i == c.i);
    assert(unboxed.i == iface.i);    
}

function object_any()
{
    const c = {
        i: 10,
        show() {
            print(this.i);
        }
    };

    const box = c as any;

    const unboxed = box as typeof c;

    assert(unboxed.i == c.i);

    unboxed.show();    
}

function main() {

    class_any();
    array_any();
    iface_any();
    object_any();

    print("done.");
}
