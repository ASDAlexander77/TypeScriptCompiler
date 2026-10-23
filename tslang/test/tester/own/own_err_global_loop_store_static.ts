// -mm=own rejects: `K.w`'s one store is in a loop of its static block, so it runs once per round and
// `K.w` is not immortal (spec 25): each store frees the value before it. `f` returns a borrow of
// the global.
class V {
    constructor(public x: number) {}
}

class K {
    static w: V;
    static {
        for (let i = 0; i < 10; i++) {
            K.w = new V(i + 1);
        }
    }
}

function f() {
    return K.w;
}

function main() {
    print(f().x);
}
