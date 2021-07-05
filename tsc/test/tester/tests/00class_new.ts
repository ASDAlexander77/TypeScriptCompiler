class c1 {
    pin: number;

    hello() {
        this.#hello();
    }

    #hello() {
        print("Hello World", this.pin);
        this.pin = 20;
    }
}

function main() {
    const c = new c1();
    c.pin = 10;
    c.hello();
    print("Hello World", c.pin);
    delete c;

    print("done.");
}
