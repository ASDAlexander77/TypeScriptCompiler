class A {

    private data: number | null = 10;

    constructor() {
        print("A");
    }

    get x(): number | null {
        return 1;
    }

    test() {
        if (this.data !== null) {
            print(this.data);
        }
    }
}

const a = new A();

if (a.x !== null) {
    print(a.x);
}

a.test();

print("done.");