class c1 {
    pin: number;

    constructor(pin: number) {
        this.pin = pin;
    }

    hello() {
        this.#hello();
    }

    #hello() {
        print("Hello World", this.pin);
        this.pin = 20;
    }
}

function main() {
    const c = new c1(1);
    print("Hello World", c.pin);
    assert(c.pin == 1, "constructor");
    c.pin = 10;
    c.hello();
    print("Hello World", c.pin);
    delete c;

    print("done.");
}
