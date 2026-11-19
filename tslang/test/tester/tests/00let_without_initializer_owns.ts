// A `let` of a heap type declared without an initializer owns what it is given, as one declared with
// a value does. Under -mm=rc it borrowed: `failure = e` in a catch clause took no reference, the clause
// gave the caught value back as it ended, and `throw failure` after it threw a freed block (the default
// library's DisposableStack.dispose; the wasm host's heap check found the second free).
class Failure {
    constructor(public name: string) {
    }
}

function run(f: () => void) {
    let failure: any;
    let failed = false;
    try {
        f();
    } catch (e) {
        failure = e;
        failed = true;
    }

    if (failed) throw failure;
}

function keep(n: number): string {
    let kept: string;
    {
        const made = "v" + n;
        kept = made;
    }

    return kept + "!";
}

function main() {
    let caught = "";
    for (let i = 0; i < 3; i++) {
        try {
            run(() => { throw new Failure("f" + i); });
        } catch (e) {
            caught += (<Failure>e).name;
        }
    }

    assert(caught == "f0f1f2", "thrown again after the catch clause");
    assert(keep(7) == "v7!", "kept past the block that made it");
    print("done.");
}
