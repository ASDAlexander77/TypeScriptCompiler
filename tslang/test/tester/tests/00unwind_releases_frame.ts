// An exception that unwinds out of a function gives back what the function holds (-mm=rc and own,
// UnwindReleasePass): its temporaries, its folded `const`s and its locals. Each throw through such a
// frame used to leak them - `const s = "x" + i; f(i);`, 2M throws: 178 MB under rc, 41 MB now.
//
// What a test can see is that the values stay right, a `catch` around still catches, a `using` around
// is still disposed and a value moved away before the throw is not freed twice; the memory is in the PR.

let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed++;
    }
}

class Holder {
    item: string = "";
}

function f(i: number): number {
    if (i >= 0) throw i;
    return 1;
}

// a temporary and a folded const held across the call that throws
function constAndTemporary(i: number) {
    const s = "x" + i;
    if (s.length > 0) f(i);
    return -1;
}

// a local
function local(i: number) {
    let s = "y" + i;
    if (s.length > 0) f(i + 1);
    return s.length;
}

// a value moved away before the call: not freed by the unwind
function movedAway(i: number, h: Holder) {
    const s = "z" + i;
    h.item = s;
    f(i + 2);
    return -1;
}

// a call that does not throw this time: the normal path still releases
function noThrow(i: number) {
    const s = "w" + i;
    f(-1);
    return s.length;
}

// an indirect call that may throw
function viaPointer(i: number, g: (n: number) => number) {
    const s = "v" + i;
    if (s.length > 0) g(i + 3);
    return -1;
}

// a `using` around and a `catch` around, in the caller
function caller(i: number) {
    try {
        using r = new Res();
        return constAndTemporary(i);
    } catch (e) {
        return <number>e;
    }
}

function main() {
    const n = 20000;
    const h = new Holder();
    for (let i = 0; i < n; i++) {
        let t: number = 0;
        try { constAndTemporary(i); t += 1; } catch (e) { t += <number>e - i; }
        try { local(i); t += 1; } catch (e) { t += <number>e - (i + 1); }
        try { movedAway(i, h); t += 1; } catch (e) { t += <number>e - (i + 2); }
        t += h.item == "z" + i ? 0 : 1;
        t += noThrow(i) - ("w" + i).length;
        try { viaPointer(i, f); t += 1; } catch (e) { t += <number>e - (i + 3); }
        t += caller(i) - i;
        if (t != 0) {
            print("wrong at", i, t);
            assert(false);
        }
    }

    assert(disposed == n);
    print("done.");
}
