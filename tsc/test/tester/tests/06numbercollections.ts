function testNumCollection(): void {
    print("test num coll");
    let collXYZ: number[] = [];
    assert(collXYZ.length == 0, "1");
    collXYZ.push(42);
    print("#1");
    assert(collXYZ.length == 1, "2");
    collXYZ.push(22);
    assert(collXYZ[1] == 22, "3");
    print("#2");
    //collXYZ.splice(0, 1);
    print("#2");
    //assert(collXYZ[0] == 22, "4");
    print("#2");
    //collXYZ.removeElement(22);
    collXYZ.pop();
    collXYZ.pop();
    print("#2");
    assert(collXYZ.length == 0, "5");
    print("loop");
    for (let i = 0; i < 100; i++) {
        collXYZ.push(i);
    }
    assert(collXYZ.length == 100, "6");

    //collXYZ = [1, 2, 3];
    // TODO: cast int[] -> number[]
    collXYZ = [1.0, 2.0, 3.0];
    assert(collXYZ.length == 3, "cons");
    assert(collXYZ[0] == 1, "cons0");
    assert(collXYZ[1] == 2, "cons1");
    assert(collXYZ[2] == 3, "cons2");
    print("loop done");
}

function main() {
    testNumCollection();
    print("done.");
}
