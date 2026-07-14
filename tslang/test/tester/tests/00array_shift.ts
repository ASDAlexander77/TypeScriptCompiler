function testShiftNumbers(): void {
    // regression test: ArrayShiftOp used to compute the post-shift memmove
    // offset with a fixed i32 stride instead of the element's real size,
    // corrupting arrays of wider-than-i32 elements (e.g. number/f64).
    let arr: number[] = [10.5, 20.5, 30.5, 40.5];

    let first = arr.shift();
    assert(first == 10.5, "shift return value");
    assert(arr.length == 3, "length after shift");
    assert(arr[0] == 20.5, "arr[0] after shift");
    assert(arr[1] == 30.5, "arr[1] after shift");
    assert(arr[2] == 40.5, "arr[2] after shift");

    let second = arr.shift();
    assert(second == 20.5, "second shift return value");
    assert(arr.length == 2, "length after second shift");
    assert(arr[0] == 30.5, "arr[0] after second shift");
    assert(arr[1] == 40.5, "arr[1] after second shift");
}

function main() {
    testShiftNumbers();
    print("done.");
}
