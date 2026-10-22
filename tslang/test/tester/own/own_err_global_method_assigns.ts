// -mm=own rejects: `G.m()` runs with `this` borrowed from `G`, and `m` gives `G` a new value, which
// frees the object `this` still is.
class C {
    x = 5;

    m() {
        G = new C();
        return this.x;
    }
}

let G = new C();

function main() {
    print(G.m());
}
