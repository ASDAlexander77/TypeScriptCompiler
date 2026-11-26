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

// - strtod declared (as the default library declares it) and a string converted to a number: the conversion's own
//   strtod was a second symbol of that name ("redefinition of symbol named 'strtod'", a crash before)
@nodrops declare function strtod(str: string, end: Reference<string> | null): double;

function toNumbers(x: any) {
    const arr: number[] = ["3", x];
    return arr[0];
}

function main() {
    assert(total == 20, "classes in blocks of the module's code");
    assert(new K().v == 3, "a class expression of the module's code");
    assert(toNumbers(1) == 3 && strtod("2.5", null) == 2.5, "strtod declared and used by a conversion");
    print("done.");
}
