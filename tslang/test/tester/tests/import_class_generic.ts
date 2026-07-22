import './export_class_generic'

function main() {
    const b = new M.Box<number>(10);
    assert(b.get() == 10);
    b.set(20);
    assert(b.get() == 20);

    const bs = new M.Box<string>("hi");
    assert(bs.get() == "hi");
    bs.set("bye");
    assert(bs.get() == "bye");

    const p = new M.Pair<number, string>(1, "one");
    assert(p.swapDescribe() == "one-1");

    print("done.");
}
