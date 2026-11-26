// TSLANG 197: classes declared in blocks of the module's code, and class expressions of it, are discovered where they
// are made (a member reading the block's locals is an error: class_reads_block_local_error.ts); those that read none
// work as before
let total: number = 0;
for (let i = 0; i < 3; i++) {
    class C {
        f = 2;
        m(k: number) {
            return this.f * k;
        }
        static s() {
            return 1;
        }
        get g() {
            return this.f;
        }
    }

    total += new C().m(i) + C.s() + new C().g;
}

{
    class D {
        v = 5;
    }

    total += new D().v;
}

const K = class {
    v = 3;
};

function main() {
    assert(total == 20, "classes in blocks of the module's code");
    assert(new K().v == 3, "a class expression of the module's code");
    print("done.");
}
