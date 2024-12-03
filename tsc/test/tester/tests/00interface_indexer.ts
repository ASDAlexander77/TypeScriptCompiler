class Test {

    #val: string;

    // to be able to assign get/set to xxx[number]
    [index1: number]: string;
    
    get(index: number): string {
        return this.#val;
    }

    set(index: number, value: string) {
        // nothing todo
        this.#val = value;
    }

    get val(): string {
        return "Hello - val";
    }

    public get_val2(): string {
        return "Hello - get_val2";
    }    
}

interface ITest {
    [index1: number]: string;

    get val(): string;

    get_val2(): string;
}

const t = new Test();

t[10] = "hello - indexer - class";

print(t[10]);

assert(t[10] == "hello - indexer - class");

print(t.get_val2());

assert(t.get_val2() == "Hello - get_val2");

const ti: ITest = t;

ti[10] = "hello - indexer - interface";

print(ti[10]);

assert(ti[10] == "hello - indexer - interface");

print(ti.get_val2());

assert(ti.get_val2() == "Hello - get_val2");

print(ti.val);

assert(ti.val == "Hello - val");

{
    print("done.");
}
