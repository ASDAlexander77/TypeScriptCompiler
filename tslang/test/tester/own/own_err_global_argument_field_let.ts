// -mm=own rejects: `x` borrows the field `h.c` of a parameter that is `G`'s object, and `run` gives
// `G` a new value before `x` is read, which frees the object and its field with it.
class C {
    x = 5;
}

class H {
    c: C | null = null;
}

let G = new H();

function run(h: H) {
    h.c = new C();
    const x = h.c;
    G = new H();
    print(x.x);
}

function main() {
    run(G);
}
