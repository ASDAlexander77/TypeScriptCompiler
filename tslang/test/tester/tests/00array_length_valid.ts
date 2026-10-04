// Array lengths that are valid keep working with the checks of #483 in place.
type Numbers = number[];

function main() {
    const a: number[] = [1, 2, 3];
    let n: number = 5;
    a.length = n;
    assert(a.length == 5 && a[4] == 0, "a number length");
    a.length = 1;
    assert(a.length == 1 && a[0] == 1, "shrink");
    for (let i = 0; i < 100; i++) a.push(i);
    assert(a.length == 101 && a[100] == 99, "growth");
    let k = 3;
    const b = new Numbers(k);
    assert(b.length == 3 && b[2] == 0, "a new array of a length");
    a.length = 0;
    assert(a.length == 0, "empty");
    print("done.");
}
