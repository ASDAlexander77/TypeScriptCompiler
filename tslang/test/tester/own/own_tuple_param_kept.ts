// -mm=own: a tuple made of borrowed strings stored into an element - also cast to a named tuple,
// which reads its fields out into a new one - starts from copies of them (spec 22). The function
// keeps nothing: its callers keep their strings, and may give the same one to it twice.
type Entry = [key: string, n: number];

class Table {
    entries: Entry[] = [];

    put(key: string, n: number) {
        this.entries.length = this.entries.length + 1;
        this.entries[this.entries.length - 1] = <Entry>[key, n];
    }
}

function putBoth(a: Table, b: Table, k: string) {
    a.put(k, 1);
    b.put(k, 2);
}

function main() {
    const t = new Table();
    const keys = ["a" + 1];
    t.put(keys[0], 7);
    keys[0] = "zzzzzzzzzzzzzzzz";
    t.put("b" + 2, 8);
    const k = "c" + 3;
    t.put(k, 9);
    const u = new Table();
    putBoth(t, u, k);
    assert(t.entries[0][0] == "a1" && t.entries[0][1] == 7 && t.entries[1][0] == "b2" && t.entries[2][0] == "c3");
    assert(t.entries[3][0] == "c3" && u.entries[0][0] == "c3" && k == "c3" && keys[0] == "zzzzzzzzzzzzzzzz");
    print("done.");
}
