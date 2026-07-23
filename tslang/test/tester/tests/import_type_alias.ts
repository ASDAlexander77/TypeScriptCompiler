import './export_type_alias'

function main() {
    const id: M.Id = 42;
    assert(id == 42);

    const uid: M.UserId = 7;
    assert(uid == 7);

    const p: M.Point = M.makePoint(1, 2);
    assert(p.x == 1);
    assert(p.y == 2);

    let s: M.Status = 5;
    assert(s == 5);
    s = "error";
    assert(s == "error");

    print("done.");
}
