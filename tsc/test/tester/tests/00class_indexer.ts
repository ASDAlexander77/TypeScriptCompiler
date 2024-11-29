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
}

const t = new Test();

t[10] = "hello";

assert(t[10] = "hello");

print(t[10]);

{
    print("done.");
}
