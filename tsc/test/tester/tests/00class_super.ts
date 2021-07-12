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
        super(name); // call the super class constructor and pass in the name parameter
    }

    speak() {
        print(`${this.name} barks.`);
    }
}

class Cat {
    name: string;

    constructor(name: string) {
        this.name = name;
    }

    speak() {
        print(`${this.name} makes a noise.`);
    }
}

class Lion extends Cat {
    speak() {
        super.speak();
        print(`${this.name} roars.`);
    }
}

function main() {
    let d = new Dog("Mitzie");
    d.speak(); // Mitzie barks.

    assert(d.name == "Mitzie");
    assert(d.super.name == "Mitzie");

    let l = new Lion("Fuzzy");
    l.speak();

    print("done.");
}
