class c1 {
    static #PRIVATE_STATIC_FIELD: number;
    #pin: number;

    static hello() {
        print("Hello World");
    }
}

namespace nn {
    class c1 {
        static #PRIVATE_STATIC_FIELD: number;
        #pin: number;

        static hello() {
            print("nn.Hello World");
        }
    }
}

function main() {
    const c = new c1();
    c.#pin = 10;
    print(c.#pin);

    c1.#PRIVATE_STATIC_FIELD = 20;
    print(c1.#PRIVATE_STATIC_FIELD);

    c1.hello();

    nn.c1.#PRIVATE_STATIC_FIELD = 30;
    print(c1.#PRIVATE_STATIC_FIELD);
    print(nn.c1.#PRIVATE_STATIC_FIELD);

    nn.c1.hello();

    delete c;

    print("done.");
}
