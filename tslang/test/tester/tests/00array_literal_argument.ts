// An array literal given straight to a call. Under -mm=own nothing gave back one of new objects (`f([new C()])`),
// nor one of a new value and a named string (`f(["x" + i, s])`, which is given a copy of `s`): both are owned
// now, released after the call or moved into a callee that keeps them. And what a call in a `try` of a loop held
// when it threw - the literal, a temporary string - was not given back when the catch was in the same function
// (the handler goes back round the loop, where the same code makes the next turn's values: taken for a use of
// this turn's), under -mm=rc as well. The values are what is checked here; the memory was measured (TSLANG 161).
class C {
    constructor(public n: string) {
    }
}

class Keeper {
    items: C[] = [];

    keep(a: C[]) {
        this.items = a;
    }
}

function read(a: C[]) {
    return a[0].n.length + a.length;
}

function firstName(a: C[]) {
    return a[0].n;
}

function popOne(a: C[]) {
    const c = a.pop();
    return c ? c.n.length : 0;
}

function pick(a: C[]) {
    return a[1];
}

function mixed(a: string[]) {
    return a[0] + a[1];
}

function failing(a: C[]): number {
    if (a.length > 1) {
        throw "failed " + a[0].n;
    }

    return 1;
}

function failingText(s: string): number {
    if (s.length > 0) {
        throw "failed " + s;
    }

    return 1;
}

function main() {
    const k = new Keeper();
    let caught = 0;
    for (let i = 0; i < 1000; i++) {
        assert(read([new C("a" + i), new C("b" + i)]) == ("a" + i).length + 2);

        k.keep([new C("k" + i), new C("l" + i)]);
        assert(k.items.length == 2 && k.items[1].n == "l" + i);

        assert(firstName([new C("f" + i), new C("g" + i)]) == "f" + i);
        assert(popOne([new C("p" + i), new C("q" + i)]) == ("q" + i).length);

        const second = pick([new C("s" + i), new C("t" + i)]);
        assert(second.n == "t" + i);

        const named = "n" + i;
        assert(mixed(["x" + i, named]) == "x" + i + "n" + i);
        assert(mixed([named, named]) == named + named && named == "n" + i);

        try {
            caught += failing([new C("t" + i), new C("u" + i)]);
        } catch (e) {
            caught += 2;
        }

        try {
            caught += failingText("x" + i);
        } catch (e) {
            caught += 3;
        }
    }

    assert(caught == 5000 && k.items[0].n == "k999");
    print("done.");
}
