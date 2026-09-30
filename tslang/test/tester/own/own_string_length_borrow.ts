// -mm=own: reading a string's length borrows the string. `box.text.length` - the field's string,
// read and let go - was rejected as the field's borrow escaping, because `ts.StringLength` was not
// among the uses own knows keep nothing.
class Box {
    text: string;
    constructor(t: string) {
        this.text = t;
    }
}

function lengthOf(box: Box) {
    return box.text.length;
}

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        const box = new Box("n" + i);
        total += lengthOf(box);
        const s = "x" + i;
        total += s.length;
    }

    // "n0".."n999" and "x0".."x999": 1000 prefixes and 10 * 1 + 90 * 2 + 900 * 3 digits, twice
    assert(total == 2 * (1000 + 2890));
    print("done.");
}
