import './export_class_static_instance_same_name'

function main() {
    const r = new M.Range(1, 5);
    assert(r.contains(1) && r.contains(5), "the instance method: both ends in");
    assert(!M.Range.contains(r, 1) && M.Range.contains(r, 3), "the static method: the ends out");
    assert(r.kind == "closed" && M.Range.kind == "Range", "a static and an instance getter");

    assert(M.Range.made == 0, "a static getter");
    M.Range.made = 7;
    assert(M.Range.made == 7, "a static setter");

    print("done.");
}
