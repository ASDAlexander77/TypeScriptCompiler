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

function t_undef_undef(s?: string, s2?: string) {
    assert((s == s2) == true, "is not undefined");
    assert((s != s2) == false, "is undefined");
    assert(s > s2 == false, "not >");
    assert(s < s2 == false, "not <");
    assert(s >= s2 == true, "not >=");
    assert(s <= s2 == true, "not <=");
}

function t_val_undef(s?: string, u?: string) {
    assert((s == u) == false, "is undefined");
    assert((s != u) == true, "is not undefined");
    assert(s > u == true, ">");
    assert(s < u == false, "<");
    assert(s >= u == true, ">=");
    assert(s <= u == false, "<=");
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
    t_undef();
    t_val("asd");

    t_undef_undef();
    t_val_undef("asd");

    class_iface();

    print("done.");
}
