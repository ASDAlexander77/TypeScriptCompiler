// regression test for #238: getFieldTypeByFieldName's ClassType branch looked up
// class field/method info in the *interface* registry (getInterfaceInfoByFullName)
// instead of the class registry (getClassInfoByFullName). Since classes and
// interfaces are registered in separate tables, this lookup silently failed for
// every field/method on a real class -- breaking both the `in` operator and
// structural generic constraints (`T extends { length: number }`) matched against
// a class instance.

class Box {
    length: number;

    constructor(length: number) {
        this.length = length;
    }
}

function getLength<T extends { length: number }>(x: T): number {
    return x.length;
}

function main() {
    const b = new Box(42);

    assert("length" in b);

    assert(getLength(b) == 42);

    print("done.");
}
