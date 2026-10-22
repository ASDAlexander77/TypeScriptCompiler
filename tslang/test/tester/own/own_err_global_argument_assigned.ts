// -mm=own rejects: `run` borrows `G`'s object as `h`, then gives `G` a new value, which frees that
// object while `h` still reads it.
class C {
    x = 5;
}

function mk(): C {
    return new C();
}

let G: C = mk();

function run(h: C) {
    G = mk();
    print(h.x);
}

function main() {
    run(G);
}
