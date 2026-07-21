import './export_class_extends'

class Dog extends M.Animal {
    constructor(name: string) {
        super(name);
    }

    speak(): string {
        return `${this.name} barks.`;
    }
}

function main() {
    const a = new M.Animal("Generic");
    assert(a.speak() == "Generic makes a noise.");

    const d = new Dog("Mitzie");
    assert(d.name == "Mitzie");
    assert(d.speak() == "Mitzie barks.");

    // virtual dispatch through the base-class-typed reference must still
    // resolve to the cross-module-derived override
    const asBase: M.Animal = d;
    assert(asBase.speak() == "Mitzie barks.");

    print("done.");
}
