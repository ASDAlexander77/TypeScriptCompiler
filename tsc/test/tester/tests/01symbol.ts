function testToPrimitive() {

    const object1 = {

        [Symbol.toPrimitive](hint: string) : string | number | boolean {
            if (hint === "number") {
                return 10;
            }
            if (hint === "string") {
                return "hello";
            }
            return true;
        }

    };

    print(+object1); // 10        hint is "number"
    print(`${object1}`); // "hello"   hint is "string"
    print(object1 + ""); // "true"    hint is "default"

    assert(+object1 == 10);

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