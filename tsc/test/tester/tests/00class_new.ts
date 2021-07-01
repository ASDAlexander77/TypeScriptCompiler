class c1 {
    #pin: number;
}

function main() {
    const c = new c1();
    c.#pin = 10;
    print(c.#pin);

    delete c;

    print("done.");
}
