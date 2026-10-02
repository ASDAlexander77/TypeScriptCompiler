// A write into a copy's bytes changes the copy only (spec 22.1): strings are values under own.
class R {
    s = "";
    t = "";
}

function main() {
    const r = new R();
    r.s = "abc" + "";
    r.t = r.s;
    r.t[0] = <char>65;
    assert(r.t == "Abc", "the copy was written");
    assert(r.s == "abc", "the original was not");

    print("done.");
}
