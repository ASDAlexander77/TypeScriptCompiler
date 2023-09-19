class c1 {
    pin: number;

    constructor(pin: number) {
        this.pin = pin;
    }

    hello() {
        return this.#hello();
    }

    #hello() {
        return this.pin;
    }
}

function main() {
    const c = new c1(2);
    print("Hello World", c.hello());
    assert(c.hello() == 2);
    delete c;

    print("done.");
}
