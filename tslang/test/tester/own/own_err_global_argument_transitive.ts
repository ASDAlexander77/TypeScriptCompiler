// -mm=own rejects: `run` borrows `G`'s object as `h` and calls `reset`, which gives `G` a new value:
// that frees the object while `h` still reads it. The callee drops it through another call.
class C {
    x = 5;
}

function mk(): C {
    return new C();
}

let G: C = mk();

function reset() {
    G = mk();
}

function run(h: C) {
    reset();
    print(h.x);
}

function main() {
    run(G);
}
