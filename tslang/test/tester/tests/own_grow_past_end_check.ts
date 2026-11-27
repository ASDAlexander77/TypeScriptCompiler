// TSLANG 152, -mm=own: an element stored past the end through a local over a parameter's array (`let src = order;
// src[n] = v`) is not grown under -mm=own yet; it was written past the end of the block (a corrupted heap). Now a
// RangeError (an assert without the default library); a store inside the array is unchanged.
function fill(order: number[], n: number) {
    let src = order;
    src[n] = 7;
}

function main() {
    const a: number[] = [1, 2];
    fill(a, 1);
    assert(a[1] == 7, "inside");
    print("inside done.");
    fill(a, 3);
    print("unreachable");
}
