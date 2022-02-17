function toLength(this: string[]) {
    return this.length;
}

function main() {
    let arr = ["asd", "asd2"];
    print(arr.toLength());
    assert(arr.toLength() == 2);
    print("done.");
}