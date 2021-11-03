function t_undef(s?: string) {
    assert((s == undefined) == true, "is not undefined");
    assert((s != undefined) == false, "is undefined");
    assert(s > undefined == false, "not >");
    assert(s < undefined == false, "not <");
    assert(s >= undefined == true, "not >=");
    assert(s <= undefined == true, "not <=");
}

function t_val(s?: string) {
    assert((s == undefined) == false, "is undefined");
    assert((s != undefined) == true, "is not undefined");
    assert(s > undefined == true, ">");
    assert(s < undefined == false, "<");
    assert(s >= undefined == true, ">=");
    assert(s <= undefined == false, "<=");
}

function f(s?: string) {
    print(
        s == undefined,
        s != undefined,
        s > undefined,
        s < undefined,
        s >= undefined,
        s <= undefined
    );
}

interface IFace {
}

class C1 {
}

function class_iface() {
    const i: IFace = undefined;
    assert(i === undefined);
    assert(!(i !== undefined));

    const c: C1 = undefined;
    assert(c === undefined);
    assert(!(c !== undefined));
}

function main() {
    f();
    t_undef();
    f("asd");
    t_val("asd");

    class_iface();

    print("done.");
}
