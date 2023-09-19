function testNumCollection(): void {
    let collXYZ: number[] = [];
    assert(collXYZ.push(42) == 1, "push count");
    assert(collXYZ.length == 1, "length");
    assert(collXYZ[0] == 42, "value");

    assert(collXYZ.pop() == 42, "pop value");
    assert(collXYZ.length == 0, "length after pop");
}

function main() {
    testNumCollection();
    print("done.");
}
