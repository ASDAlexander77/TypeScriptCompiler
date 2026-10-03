// A borrowed array parameter is the owner's array: a push through it is the owner's (#453).
function keep(into: number[]) {
    into.push(2);
    into.push(3);
}

function main() {
    let a: number[] = [7];
    keep(a);
    assert(a.length == 3 && a[0] == 7 && a[2] == 3, "push through a borrowed parameter");
    print("done.");
}
