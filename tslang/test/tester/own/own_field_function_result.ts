// What a call through a field's function returns is the caller's own, not a part of the function
// or of the object holding it: a string kept after the object is gone, read through a field typed
// as a function, an arrow field reading `this`, and a function value read out of the field first.
// Each step churns, so anything freed too early reads garbage; every model runs it.
class Junk {
    constructor(public n: number) {}
}

function churn() {
    const junk: Junk[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Junk(999));
}

class A {
    name = "a";
    typed: () => string = () => "typed";
    arrow = () => this.name + "!";
}

function make(name: string) {
    const a = new A();
    a.name = name;
    return a;
}

function kept() {
    const a = make("b");
    return a.arrow();
}

function main() {
    const a = new A();
    const s = a.typed();
    const t = a.arrow();
    const f = a.typed;
    const u = f();
    churn();
    assert(s == "typed" && t == "a!" && u == "typed", "results of calls through fields");

    const k = kept();
    churn();
    assert(k == "b!", "a result kept after its object is released");

    let all = "";
    for (let i = 0; i < 10; i++) {
        all = all + make("x").arrow();
    }

    churn();
    assert(all.length == 20, "results of temporaries' fields");

    print("done.");
}
