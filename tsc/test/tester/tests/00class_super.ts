class Animal {
    name: string;

    constructor(name: string) {
        this.name = name;
    }

    speak() {
        print(`${this.name} makes a noise.`);
    }
}

class Dog extends Animal {
    constructor(name: string) {
        //super(name);
        print("in constructor");
        this.Animal.name = name;
        super.name = name;
        print("end constructor");
    }

    speak2() {
        print("in speak2");
        print(`${this.Animal.name} test`);
        print(`${super.name} test`);
        this.Animal.speak();
        super.speak();
        print("end speak2");
    }
}

function main() {
    let d = new Dog("Mitzie");
    d.speak(); // Mitzie barks.
    d.speak2(); // Mitzie barks.
    print(d.super.name);
    print("done.");
}
