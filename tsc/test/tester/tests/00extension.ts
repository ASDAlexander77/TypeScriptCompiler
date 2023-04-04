function toLength(this: string[]) {
    return this.length;
}

function toLength2<T>(this: T[]) {
    return this.length;
}

function toLength3<T, V>(this: T[], v: V) {
    print("v:", v);
    return this.length;
}

function main() {
    let arr = ["asd", "asd2"];
    print(arr.toLength());
    assert(arr.toLength() == 2);

    print(arr.toLength2());
    assert(arr.toLength2() == 2);

    print(arr.toLength3(10));
    assert(arr.toLength3(10) == 2);

    let arrInt = [1, 2];

    print(arrInt.toLength2());
    assert(arrInt.toLength2() == 2);

    print(arrInt.toLength3(10));
    assert(arrInt.toLength3(10) == 2);

    print("done.");
}