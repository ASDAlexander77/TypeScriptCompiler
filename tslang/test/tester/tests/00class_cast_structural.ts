// Classes where neither extends the other are compatible by their fields, as in TypeScript.
class Point {
    constructor(public x: number, public y: number) {}
}

class Vec {
    constructor(public x: number, public y: number) {}
}

class Named {
    name = "n";
}

class Dog {
    bark() {
        return 1;
    }
}

class Cat {
    meow() {
        return 2;
    }
}

function isCat(p: Dog | Cat): p is Cat {
    return p instanceof Cat;
}

function main() {
    // the same fields
    const p = new Point(1, 2);
    const v: Vec = p;
    assert(v.x == 1 && v.y == 2, "same shape");

    // no fields on either side
    const d = new Dog();
    if (isCat(d)) {
        assert(false, "a dog is not a cat");
    }

    const n = new Named();
    assert(n.name == "n", "named");

    print("done.");
}
