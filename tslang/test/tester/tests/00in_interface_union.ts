// `"k" in a` and `a instanceof I` for a union of interfaces. Which members have the key, or are the
// interface, is known when compiling; which member the value holds is asked of the union's tag, which
// says "interface" for every one but whose descriptor names which. `in` narrows the value to the members
// with the key, and the `else` to the others. Before, `in` on a union was "not supported" and
// `a instanceof Fish` was always false.
interface Fish {
    swim(): string;
}

interface Bird {
    fly(): string;
}

interface Duck {
    swim(): string;
    fly(): string;
    quack(): string;
}

class Salmon implements Fish {
    swim() { return "swims"; }
}

class Crow implements Bird {
    fly() { return "flies"; }
}

class Mallard implements Duck {
    swim() { return "paddles"; }
    fly() { return "flaps"; }
    quack() { return "quacks"; }
}

function move(a: Fish | Bird) {
    if ("swim" in a) {
        return a.swim();
    } else {
        return a.fly();
    }
}

function moveAll(a: Fish | Bird | Duck) {
    if ("quack" in a) {
        return a.quack();
    }

    if ("swim" in a) {
        return "swim:" + a.swim();
    }

    return "fly";
}

function kind(a: Fish | Bird) {
    if (a instanceof Fish) {
        return "fish";
    }

    return "bird";
}

function main() {
    const f: Fish = new Salmon();
    const b: Bird = new Crow();
    const d: Duck = new Mallard();

    assert(move(f) == "swims");
    assert(move(b) == "flies");

    assert(moveAll(f) == "swim:swims");
    assert(moveAll(b) == "fly");
    assert(moveAll(d) == "quacks");

    assert(kind(f) == "fish");
    assert(kind(b) == "bird");

    // a key every member has, or none: decided when compiling
    const both: Fish | Duck = d;
    assert("swim" in both);
    assert(!("bark" in both));

    print("done.");
}
