// TSLANG 168: a class given to a generic constructor type (`new (n: number) => T`), and a module `unknown` given a
// value under every model

class Animal {
    constructor(public name: string) {}
    speak() { return this.name + " makes a sound"; }
}

class Dog extends Animal {
    speak() { return this.name + " barks"; }
}

class Point {
    x = 1;
    y = 2;
}

function create<T>(ctor: new () => T): T { return new ctor(); }
function named<T extends Animal>(ctor: new (n: string) => T, n: string): T { return new ctor(n); }
function many<T>(ctor: new (n: string) => T, count: number): T[] {
    const r: T[] = [];
    for (let i = 0; i < count; i++) r.push(new ctor("a" + i));
    return r;
}

let u: unknown = 7.5;
let s: unknown = "str" + 1;

function main() {
    const p = create(Point);
    assert(p.x + p.y == 3);
    assert(named(Dog, "rex").speak() == "rex barks");
    assert(named(Animal, "cat").speak() == "cat makes a sound");

    const dogs = many(Dog, 3);
    assert(dogs.length == 3);
    assert(dogs[2].speak() == "a2 barks");

    let total = 0;
    for (let i = 0; i < 1000; i++) total += named(Dog, "d" + i).name.length;
    assert(total == 3890);

    let n: number = 0;
    if (typeof u === "number") n = u + 1;
    assert(n == 8.5);
    let text = "";
    if (typeof s === "string") text = s;
    assert(text == "str1");
    u = "x" + 2;
    if (typeof u === "string") text = u;
    assert(text == "x2");

    print("done.");
}