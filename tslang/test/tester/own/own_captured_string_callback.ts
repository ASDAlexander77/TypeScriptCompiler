// A callback that assigns a captured string (`each += v`) frees only the string the variable held
// (spec 52): a call that runs it, given the array a field holds, drops nothing that borrow reads.
// Each step churns, so an array freed too early reads garbage; every model runs it.
class C {
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) junk.push(new C(999));
}

function label(i: number) {
    return "s" + i;
}

// a function of its own, not the built-in `forEach`, which is inlined without the default library
function forAll(a: string[], func: (v: string) => void) {
    for (let i = 0; i < a.length; i++) func(a[i]);
}

class Names {
    constructor(private data: string[] = []) {}

    forEach(func: (v: string) => void) {
        forAll(this.data, func);
    }

    count(func: (v: string) => boolean) {
        let n = 0;
        for (const v of this.data) if (func(v)) n++;
        return n;
    }
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const names = new Names([label(1), label(22), label(333)]);
        let each = "";
        names.forEach((v) => {
            each += v;
        });
        churn();
        if (each == "s1s22s333") ok++;

        let seen = "";
        const n = names.count((v) => {
            seen = v;
            return v.length > 2;
        });
        churn();
        if (n == 2 && seen == "s333") ok++;
    }

    assert(ok == 300 * 2, "ok");
    print("done.");
}
