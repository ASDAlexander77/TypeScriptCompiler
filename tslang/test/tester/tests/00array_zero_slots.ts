// Zeroed memory is an empty array: nested arrays made by length, and a field with no initializer.
class Bag {
    items: number[];
    add(v: number) { this.items.push(v); }
}

function main() {
    // `new Array<number[]>(2)` needs the default library; growing by `length =` zeroes the same way
    const aa: number[][] = [];
    aa.length = 2;
    assert(aa[0].length == 0, "a zeroed element reads as empty");
    aa[0].push(1);
    aa[0].push(2);
    assert(aa[0].length == 2 && aa[0][1] == 2 && aa[1].length == 0, "push into a zeroed element");

    const b = new Bag();
    b.add(5);
    assert(b.items.length == 1 && b.items[0] == 5, "push into a field with no initializer");

    print("done.");
}
