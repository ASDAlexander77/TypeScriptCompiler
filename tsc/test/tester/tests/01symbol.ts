function testToPrimitive() {

    const object1 = {

        [Symbol.toPrimitive](hint: string) {
            if (hint === 'number') {
                return 42;
            }

            return null;
        }

    };

    print(+object1);

    assert(+object1 == 42);
}

class Array1 {
    static [Symbol.hasInstance](instance) {
        if (typeof instance === "string")
            print("str: ", instance);
        if (typeof instance === "number")
            print("num: ", instance);
        return false;
    }
}

function main()
{
    testToPrimitive();

    print([] instanceof Array1);

    print("done.");
}