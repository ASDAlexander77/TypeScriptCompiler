// `x[i] = v` whose right side gives the place another array - through itself, through a field, or in a call:
// the store goes into the array the left side read, and that one grows (as in JavaScript); the place keeps the
// new one. Under -mm=rc the array read is kept alive until the store (the right side released it). -mm=own
// refuses it: the left side borrows what the right side releases.

class H {
    a: number[] = [1, 2, 3];
}

function replace(h: H) {
    h.a = [7];
    return 8;
}

function main() {
    const h = new H();
    const old = h.a;
    h.a[0] = (h.a = [5], 9);
    assert(old.length == 3 && old[0] == 9, "the old array written");
    assert(h.a.length == 1 && h.a[0] == 5, "the place holds the new one");

    const k = new H();
    const old2 = k.a;
    k.a[5] = (k.a = [7], 8);
    assert(old2.length == 6 && old2[5] == 8, "the old array grown");
    assert(k.a.length == 1 && k.a[0] == 7, "the new one untouched");

    const m = new H();
    const old3 = m.a;
    m.a[4] = replace(m);
    assert(old3.length == 5 && old3[4] == 8, "a call that replaces it");
    assert(m.a.length == 1 && m.a[0] == 7, "the call's array untouched");

    let a = [1, 2];
    const old4 = a;
    a[4] = (a = [3], 6);
    assert(old4.length == 5 && old4[4] == 6, "a local");
    assert(a.length == 1 && a[0] == 3, "the local's new array");

    for (let i = 0; i < 1000; i++) {
        const t = new H();
        t.a[10] = (t.a = [i], i);
        assert(t.a[0] == i, "in a loop");
    }

    print("done.");
}
