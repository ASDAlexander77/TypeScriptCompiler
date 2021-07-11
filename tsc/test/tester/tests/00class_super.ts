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
        super(name);
    }

    speak2() {
        super.speak();
    }
}

function main() {
    let d = new Dog("Mitzie");
    d.speak();
    d.speak2();
    print(d.super.name);
    assert(d.super.name == "Mitzie");
    print("done.");
}
