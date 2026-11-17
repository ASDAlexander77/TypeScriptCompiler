// A method of every class a union may hold (`o.name()` of `A | B`) is the method of the class it
// holds: the call dispatches on the union's tag. It called the first member's method for any of them.
class ConcreteA {
    name() {
        return "A";
    }

    scaled(k: number) {
        return k * 2;
    }

    touch() {
        touched += 1;
    }
}

class ConcreteB {
    name() {
        return "B";
    }

    scaled(k: number) {
        return k * 3;
    }

    touch() {
        touched += 10;
    }
}

class ConcreteC extends ConcreteB {
    name() {
        return "C";
    }
}

let touched = 0;

function nameOf(o: ConcreteA | ConcreteB) {
    return o.name();
}

function main() {
    assert(nameOf(new ConcreteA()) == "A", "A");
    assert(nameOf(new ConcreteB()) == "B", "B");

    // with an argument, run once
    let i = 1;
    const o: ConcreteA | ConcreteB = new ConcreteB();
    assert(o.scaled(i++) == 3 && i == 2, "argument run once");

    // no result
    const p: ConcreteA | ConcreteB = new ConcreteA();
    p.touch();
    o.touch();
    assert(touched == 11, "void methods");

    // three members
    const all: (ConcreteA | ConcreteB | ConcreteC)[] = [new ConcreteA(), new ConcreteB(), new ConcreteC()];
    let names = "";
    for (const x of all) {
        names += x.name();
    }

    assert(names == "ABC", "three members");

    // classes as values, made and called
    let made = "";
    const classes: (typeof ConcreteA | typeof ConcreteB)[] = [ConcreteA, ConcreteB];
    for (const c of classes) {
        made += new c().name();
    }

    assert(made == "AB", "made and called");

    print("done.");
}
