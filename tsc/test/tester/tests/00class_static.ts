class c1 {
    static #PRIVATE_STATIC_FIELD: number;
    #pin: number;
}

namespace nn {
    class c1 {
        static #PRIVATE_STATIC_FIELD: number;
        #pin: number;
    }
}

function main() {
    const c = new c1();
    c.#pin = 10;
    print(c.#pin);

    c1.#PRIVATE_STATIC_FIELD = 20;
    print(c1.#PRIVATE_STATIC_FIELD);

    nn.c1.#PRIVATE_STATIC_FIELD = 30;
    print(c1.#PRIVATE_STATIC_FIELD);
    print(nn.c1.#PRIVATE_STATIC_FIELD);

    delete c;

    print("done.");
}
