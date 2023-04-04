function toLength2<T>(this: T[]) {
    return this.length;
}

function main() {
    let arr = ["asd", "asd2"];

    print(arr.toLength2());
    assert(arr.toLength2() == 2);

    print("done.");
}