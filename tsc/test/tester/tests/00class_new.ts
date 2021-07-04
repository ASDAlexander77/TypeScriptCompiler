class c1 {
    pin: number;

    hello() {
        print("Hello World", this.pin);
    }
}

function main() {
    const c = new c1();
    c.pin = 10;
    c.hello();
    delete c;

    print("done.");
}
