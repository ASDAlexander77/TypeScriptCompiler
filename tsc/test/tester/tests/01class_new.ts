interface Boolean {
    /** Returns the primitive value of the specified object. */
    valueOf(): boolean;
}

interface BooleanConstructor {
    new(value?: boolean): Boolean;
}

class BooleanImpl implements BooleanConstructor {
    value: boolean;

    constructor(value?: boolean) {
        this.value = value;
    }

    valueOf(): boolean {
        return this.value;
    }
}

function main() {
    const Boolean: BooleanConstructor = new BooleanImpl();
    const b = new Boolean(true);
    print(b.valueOf());

    assert(b.valueOf());

    print("done.");
}