// TSLANG 177: a name that starts with `__` is that name wherever it is declared or used (a field `__p` was declared
// as `___p` and read as `__p`: "can't resolve property"; `keyof` gave `"___p"`), and a program's own `__unbox`
// is not the compiler's helper, which is named `.unbox`

class A {
    __p = 7;
    ___q = "q";
    get __g() {
        return this.__p + 1;
    }
}

interface I {
    __i: number;
}

function __unbox(x: number) {
    return x * 10;
}

const __top = 3;

function main() {
    const o = new A();
    assert(o.__p == 7 && o["__p"] == 7 && o.___q == "q" && o.__g == 8);

    const k: keyof A = "__p";
    assert(o[k] == 7);

    let keys = "";
    for (const key in o) {
        keys = keys + key + ",";
    }

    assert(keys == "__p,___q,");

    const i: I = { __i: 5 };
    assert(i.__i == 5);

    const { __p } = o;
    assert(__p == 7);

    assert(__unbox(2) == 20 && __top == 3);

    const a: any = 5;
    assert(<number>a + 1 == 6);

    print("done.");
}
