// A global's one store that runs more than once - in a loop of a static block, or of top-level
// script code - leaves the global mortal under -mm=own (spec 25): each store releases the value it
// replaces, as for any other global. Taken as immortal, every value it was ever given would leak.
// A store that runs once, in an `if` of a static block, still makes its global immortal.
class V {
    a: number[] = [];
    constructor(public x: number) {
        for (let i = 0; i < 8; i++) {
            this.a.push(i);
        }
    }
}

@noinline
function on() {
    return true;
}

class K {
    static w: V;
    static {
        for (let i = 0; i < 10000; i++) {
            K.w = new V(i + 1);
        }
    }
}

class S {
    static w: V;
}

let i = 0;
while (i < 10000) {
    S.w = new V(i + 2);
    i++;
}

class T {
    static w: V;
    static {
        if (on()) {
            T.w = new V(3);
        }
    }
}

function t() { return T.w; }   // immortal: returned

function main() {
    print(K.w.x, K.w.a.length, S.w.x, t().x);   // 10000 8 10001 3
    assert(K.w.x == 10000 && S.w.x == 10001 && t().x == 3);
    print("done.");
}
