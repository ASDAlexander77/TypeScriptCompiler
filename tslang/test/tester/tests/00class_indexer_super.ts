class Storage {
    #val: string = "";

    [index: number]: string;

    get(index: number): string {
        return this.#val;
    }

    set(index: number, value: string) {
        this.#val = value;
    }
}

// overriding an indexer and calling back into the base implementation via
// `super[index]` / `super[index] = value` - same crash class as
// `super.<accessor>` (00class_accessor_super.ts, see the comment there):
// ClassIndexAccess never took an `isSuperClass` parameter at all (unlike
// ClassMethodAccess/ClassAccessorAccess), so `thisValue` was never repaired
// via getThisRefOfClass for `super[i]`.
class UpperStorage extends Storage {
    [index: number]: string;

    get(index: number): string {
        return super[index] + "!";
    }

    set(index: number, value: string) {
        super[index] = "U:" + value;
    }
}

function main() {
    const s = new UpperStorage();
    s[0] = "hi";
    assert(s[0] == "U:hi!");

    print("done.");
}
