// -mm=own rejects: `G.m()` runs with `this` borrowed from `G`, and `m` calls `reset`, which gives
// `G` a new value: that frees the object `this` still is.
class C {
    x = 5;

    m() {
        reset();
        return this.x;
    }
}

let G = new C();

function reset() {
    G = new C();
}

function main() {
    print(G.m());
}
