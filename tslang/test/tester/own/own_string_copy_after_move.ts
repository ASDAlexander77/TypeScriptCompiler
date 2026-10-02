// A string moved into a field and used after: the field takes a copy, the local keeps its own -
// also where the store is on one branch only, or in a loop: the copy is made beside the store.
class H {
    text = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function stored(c: boolean) {
    const h = new H();
    const s = "some" + 3;
    if (c) {
        h.text = s;
    }
    print(s);
    return h;
}

function storedInLoop() {
    const h = new H();
    const s = "loop" + 4;
    for (let i = 0; i < 3; i++) {
        h.text = s;
    }
    print(s);
    return h;
}

// A closure borrows the string and a field takes it. The closure captures a variable too, so it
// borrows all it captures: the store that fills its box borrows, and the field is the one taker,
// which gets the copy.
function captured() {
    const h = new H();
    let n = 1;
    const s = "cap" + 7;
    const f = () => {
        return s.length + n;
    };
    n = 2;
    assert(f() == 6);
    h.text = s;
    print(s);
    assert(churn() == 1000);
    assert(s == "cap7" && h.text == "cap7");
    return h;
}

function main() {
    const h = new H();
    let kept = "kept" + 1;
    h.text = kept;
    print(kept);
    h.text = "other" + 2;
    assert(churn() == 1000);
    assert(kept == "kept1" && h.text == "other2");

    const both = stored(true);
    const neither = stored(false);
    const looped = storedInLoop();
    assert(churn() == 1000);
    assert(both.text == "some3" && neither.text == "" && looped.text == "loop4");

    const closed = captured();
    assert(churn() == 1000);
    assert(closed.text == "cap7");

    print("done.");
}
