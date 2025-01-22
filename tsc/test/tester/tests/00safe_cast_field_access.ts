class A {
    constructor() {
        print("A");
    }

    get x(): number | null {
        return 1;
    }
}

const a = new A();

if (a.x !== null) {
    print(a.x);
}

print("done");