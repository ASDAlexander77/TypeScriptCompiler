import './export_interface_generic_method'

class Upper implements M.Mapper {
    map<T>(x: T): T {
        return x;
    }

    tag<T>(x: T, label: string): string {
        return "upper:" + label;
    }
}

function main() {
    const m: M.Mapper = new M.Id();
    const n: number = m.map<number>(5);
    assert(n + 1 == 6, "number");
    const s = m.map("ab");
    assert(s.length == 2, "string");
    assert(m.tag(1, "x") == "id:x", "tag");

    const u: M.Mapper = new Upper();
    assert(u.map(7) == 7, "local implementer");
    assert(u.tag(true, "y") == "upper:y", "local tag");

    assert(M.useMapper(m) == 42, "exported function, exported class");
    assert(M.useMapper(u) == 42, "exported function, local class");

    print("done.");
}
