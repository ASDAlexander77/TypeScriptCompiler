// An object literal, a literal held by a const, or a class object given to a union of interfaces becomes
// the first interface it has every member of, and the union's tag says which: `"swim" in a` then tells
// them apart. A literal was "can't be stored in" the union (a compiler error), and so was a class object.
interface Fish {
    swim(): string;
}

interface Bird {
    fly(): string;
}

class Crow implements Bird {
    fly() { return "fly"; }
}

function move(a: Fish | Bird) {
    return "swim" in a ? a.swim() : a.fly();
}

function main() {
    const k = "!";
    assert(move({ swim: () => "s" }) == "s");
    assert(move({ fly: () => "f" + k }) == "f!");
    const o = { swim: () => "o" };
    assert(move(o) == "o");
    assert(move(new Crow()) == "fly");
    print("done.");
}
