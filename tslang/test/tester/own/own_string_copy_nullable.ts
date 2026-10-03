// Nullable strings copied: a `string | null` holding a string and holding null, and an optional
// (`string | undefined`) holding a string and holding nothing - whose pointer must not be read.
class Box {
    s: string | null = null;
    o: string | undefined = undefined;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    const a = new Box();
    const b = new Box();
    a.s = "nn" + 1;
    b.s = a.s;
    a.s = null;
    assert(churn() == 1000);
    assert(b.s == "nn1");

    b.s = a.s;
    assert(b.s == null);

    a.o = "opt" + 2;
    b.o = a.o;
    a.o = undefined;
    assert(churn() == 1000);
    assert(b.o == "opt2");

    b.o = a.o;
    assert(b.o === undefined);

    print("done.");
}
