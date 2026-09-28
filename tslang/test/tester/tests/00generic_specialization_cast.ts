// A cast between two specializations of one generic class is allowed when one's type arguments
// extend the other's: up (Box<Dog> as Box<Animal>) and back down again.
class Animal {
    name = "animal";
}

class Dog extends Animal {
    bark() {
        return 1;
    }
}

class Box<T> {
    constructor(public v: T) {}
}

class Pair<A, B> {
    constructor(public a: A, public b: B) {}
}

function main() {
    const d = new Box<Dog>(new Dog());
    const up: Box<Animal> = d;
    assert(up.v.name == "animal", "up");

    const down = <Box<Dog>>up;
    assert(down.v.bark() == 1, "down");

    const p = new Pair<Dog, number>(new Dog(), 2);
    const q: Pair<Animal, number> = p;
    assert(q.b == 2, "pair");

    print("done.");
}
