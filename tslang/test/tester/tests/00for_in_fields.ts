// `for...in` over an object literal or a tuple walks the fields of its type, the key a literal, so
// `obj[key]` is that field with its own type (JSON.stringify of an object literal is made of it).

function keysOf<T>(value: T): string {
    let result = "";
    for (const key in value) {
        result += key + "=" + value[key] + ";";
    }

    return result;
}

function main() {
    const person = { name: "Ann", age: 30, nested: { a: true } };
    let keys = "";
    for (const key in person) {
        keys += key + ",";
    }

    assert(keys == "name,age,nested,", "the fields, in order");
    assert(keysOf({ x: 1, s: "a" }) == "x=1;s=a;", "each field with its own type, in a generic");

    // a tuple's fields have no names: their keys are their positions, and `t["1"]` and `"1" in t` find them
    const tuple = [1, "a"];
    assert(keysOf(tuple) == "0=1;1=a;", "a tuple's positions");
    assert(tuple["1"] == "a", "a position as a string");
    assert("0" in tuple && "1" in tuple && !("2" in tuple), "a position is in a tuple");
    assert("name" in person && !("0" in person), "a name is in an object literal, a position is not");

    // still a loop: break and continue
    let seen = "";
    for (const key in person) {
        if (key == "name") {
            continue;
        }

        if (key == "nested") {
            break;
        }

        seen += key;
    }

    assert(seen == "age", "continue and break");

    // a key declared before the loop
    let last = "";
    for (last in person) {
    }

    assert(last == "nested", "a variable declared before the loop");

    let count = 0;
    for (const key in {}) {
        count++;
    }

    assert(count == 0, "no fields");

    // an object literal with a method
    let withMethod = "";
    for (const key in { k: 2, m() { return 1; } }) {
        withMethod += key;
    }

    assert(withMethod.length == 2, "a method is a field too");

    // an accessor is one key; a value field named like an accessor's storage is its own name (the accessor
    // comes after the value fields, not where it is written)
    let accessorKeys = 0;
    let found = 0;
    for (const key in { v: 3, get x() { return 1; }, set x(n: number) { }, get_started: true }) {
        accessorKeys++;
        if (key == "v" || key == "x" || key == "get_started") {
            found++;
        }
    }

    assert(accessorKeys == 3 && found == 3, "accessors and get_ names");

    // the body is made once for each field: an arrow function in it is one for each too
    let captured = "";
    for (const key in person) {
        const keyOf = () => key;
        captured += keyOf();
    }

    assert(captured == "nameagenested", "an arrow function in the body");

    print("done.");
}
