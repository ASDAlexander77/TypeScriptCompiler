// @strict-null false
function t_null() {
    let s: string = null;
    assert((s == null) == true, "null");
    assert((s != null) == false, "!= null");
    assert(s > null == false, "> null");
    assert(s < null == false, "< null");
    assert(s >= null == true, ">= null");
    assert(s <= null == true, "<= null");
}

function t_1null() {
    let s = "asd";
    assert((s == null) == false, "null - asd");
    assert((s != null) == true, "!= null - asd");
    assert(s > null == true, "> null - asd");
    assert(s < null == false, "< null - asd");
    assert(s >= null == true, ">= null - asd");
    assert(s <= null == false, "<= null - asd");
}

function main() {
    let s: string = null;
    print(s == null);
    print(s != null);
    print(s > null);
    print(s < null);
    print(s >= null);
    print(s <= null);

    t_null();
    t_1null();

    print("done.");
}
