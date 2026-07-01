class Array1 {
    static [Symbol.hasInstance](instance: any) {
        if (typeof instance === "string")
            print("str: ", instance);
        if (typeof instance === "number")
            print("num: ", instance);
        return false;
    }
}

class ValidatorClass {
    get [Symbol.toStringTag]() {
        return 'Validator';
    }
}

interface IObj {
    [Symbol.hasInstance]: (v: any) => boolean;
}

function main() {
    Array1[Symbol.hasInstance]("hello");
    Array1[Symbol.hasInstance](<number>10);

    const obj = {
        [Symbol.hasInstance]: (instance: any) => {
            if (typeof instance === "string")
                print("obj: str: ", instance);
            if (typeof instance === "number")
                print("obj: num: ", instance);
            return true;
        }
    };

    obj[Symbol.hasInstance](<number>20);

    const iobj: IObj = obj;
    iobj[Symbol.hasInstance](<number>30);

    assert("Validator" === <string>(new ValidatorClass()));

    print("done.");
}
