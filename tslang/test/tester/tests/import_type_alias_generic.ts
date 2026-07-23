import './export_type_alias_generic'

function main() {
    const b: M.Box<number> = { value: 42 };
    assert(b.value == 42);

    const p: M.Pair<number, string> = { first: 1, second: "one" };
    assert(p.first == 1);
    assert(p.second == "one");

    print("done.");
}
