// A record variable assigned to a class type (#487): the instance is made from its fields and
// shares its arrays, which are references. Under -mm=own that is a second reference to the
// record's array, and an error; a record literal is not one (00record_to_class.ts, #492).
class P {
    x: number;
    items: number[];
    name: string;
}

function main() {
    let n: number = 5;
    let r = { x: n, items: [n, 2], name: "b" + n };
    const q: P = r;
    q.items.push(9);
    assert(r.items.length == 3 && q.name == "b5", "a record variable, sharing its array");
    print("done.");
}
