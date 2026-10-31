// A call through a class's vtable to a method no class overrides is made a direct call; one that a
// class overrides - declared before or after the call, any number of classes down - stays virtual.

// nothing extends it: its calls are direct
class Plain {
    constructor(public n: number) {}

    value() {
        return this.n * 2;
    }

    twice() {
        return this.value() + this.value();
    }
}

// `area` is overridden by a class declared after the code that calls it; `name` is not
class Shape {
    area() {
        return 0;
    }

    name() {
        return "shape";
    }
}

function areaOf(s: Shape) {
    return s.area();
}

function nameOf(s: Shape) {
    return s.name();
}

class Square extends Shape {
    constructor(public side: number) {
        super();
    }

    area() {
        return this.side * this.side;
    }
}

// `level` is overridden two classes down only
class A {
    level() {
        return 1;
    }

    kind() {
        return "a";
    }
}

class B extends A {}

class C extends B {
    level() {
        return 3;
    }
}

function levelOf(a: A) {
    return a.level();
}

function levelOfB(b: B) {
    return b.level();
}

abstract class Animal {
    abstract sound(): string;

    speak() {
        return this.sound() + "!";
    }
}

class Dog extends Animal {
    sound() {
        return "woof";
    }
}

// a specialization of a generic class, overridden by a class extending it
class Box<T> {
    constructor(public v: T) {}

    read() {
        return this.v;
    }
}

class TenBox extends Box<number> {
    read() {
        return this.v * 10;
    }
}

function readBox(b: Box<number>) {
    return b.read();
}

// overridden by a class declared inside a function
class Outer {
    m() {
        return 1;
    }
}

const makeInner = () => {
    class Inner extends Outer {
        m() {
            return 7;
        }
    }

    return new Inner();
};

function callM(o: Outer) {
    return o.m();
}

function main() {
    assert(readBox(new TenBox(2)) == 20 && readBox(new Box<number>(3)) == 3, "a generic class's override");
    assert(callM(makeInner()) == 7 && callM(new Outer()) == 1, "an override in a function's class");

    const p = new Plain(5);
    assert(p.value() == 10, "a method nothing overrides");
    assert(p.twice() == 20, "called on `this`");

    assert(areaOf(new Square(3)) == 9, "an override declared after the call");
    assert(areaOf(new Shape()) == 0, "the base's own method");
    assert(nameOf(new Square(3)) == "shape", "an inherited method");

    assert(levelOf(new C()) == 3, "overridden two classes down, through the base");
    assert(levelOfB(new C()) == 3, "overridden two classes down, through the middle class");
    assert(levelOf(new B()) == 1, "the middle class has the base's");
    assert(new C().kind() == "a", "an inherited method of a chain");

    const animal: Animal = new Dog();
    assert(animal.speak() == "woof!", "an abstract method");

    print("done.");
}
