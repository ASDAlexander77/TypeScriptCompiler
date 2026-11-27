// TSLANG 214: a `break` out of a catch clause (its temporaries given back under -mm=own), a class given to a construct
// signature whose constructor keeps a string (-mm=own: the function made for it passes a copy), and a tuple literal
// given to a union of tuples (built as the member of its length; -mm=own refused a new string in it).

class R {
    constructor(public r: string) {}
}

interface RCtor {
    new (r: string): R;
}

function makeByInterface(c: RCtor, s: string): R {
    return new c(s);
}

function makeByLiteral(c: { new (r: string): R }, s: string): R {
    return new c(s);
}

function caughtLength(k: number): number {
    let s = "";
    L: {
        try {
            throw "e" + k;
        } catch (e) {
            s = "c" + e;
            break L;
        }
    }

    return s.length;
}

function caughtInLoop(k: number): number {
    let s = "";
    for (let j = 0; j < 3; j++) {
        try {
            throw "e" + k;
        } catch (e) {
            s = "c" + e;
            break;
        }
    }

    return s.length;
}

function main() {
    let total = 0;
    for (let i = 0; i < 10; i++) {
        total += caughtLength(i) + caughtInLoop(i);
    }

    assert(total == 60);

    let made = "";
    for (let i = 0; i < 3; i++) {
        const s = "r" + i;
        made += makeByInterface(R, s).r + makeByLiteral(R, s).r;
    }

    assert(made == "r0r0r1r1r2r2");
    assert(makeByInterface(R, "lit").r == "lit");

    let lengths = 0;
    for (let i = 0; i < 3; i++) {
        const x: [number, string] | [number, string, string] = [i, "a" + i, "b"];
        lengths += x.length;
        const y: [number, string] | [number, string, string] = [i, "z" + i];
        lengths += y.length;
        let z: [number, string] | [number, string, string] = [i, "y" + i, "w" + i];
        lengths += z.length;
    }

    assert(lengths == 24);

    print("done.");
}
