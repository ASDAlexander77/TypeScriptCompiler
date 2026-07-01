class c1 {
    static #PRIVATE_STATIC_FIELD: number;
    #pin: number;

    static hello() {
        print("Hello World");
    }

    static test() {
        const c = new c1();
        c.#pin = 10;
        print(c.#pin);
    
        c1.#PRIVATE_STATIC_FIELD = 20;
        print(c1.#PRIVATE_STATIC_FIELD);
    
        c1.hello();

        delete c;
    }
}

namespace nn {
    class c1 {
        static #PRIVATE_STATIC_FIELD: number;
        #pin: number;

        static hello() {
            print("nn.Hello World");
        }

        static test() {
            nn.c1.#PRIVATE_STATIC_FIELD = 30;
            print(c1.#PRIVATE_STATIC_FIELD);
            print(nn.c1.#PRIVATE_STATIC_FIELD);
        
            nn.c1.hello();
        }
    }
}

static class Number
{
	static toString(this: number)
	{
		return `${this}`;
	}
}

function main() {
    c1.test();

    nn.c1.test();

    print(Number.toString(5.0));
    assert(Number.toString(5.0) === "5");

    print("done.");
}
