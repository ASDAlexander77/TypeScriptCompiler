let count_created = 0;
let count_disp = 0;
function loggy(id: string) {
    print(`Creating ${id}`);
    count_created++;
    return {
        [Symbol.dispose]() {
            print(`Disposing ${id}`);
	    count_disp++;
        }
    }
}

function func() {
    using a = loggy("a");
    using b = loggy("b");
    {
        using c = loggy("c");
        using d = loggy("d");
    }
    using e = loggy("e");
    return;
    // Unreachable.
    // Never created, never disposed.
    using f = loggy("f");
}

function func2(i = 0) {
    using a = loggy("a");
    if (i > 0) return;
}

function main() {
    func();
    func2();

    assert(count_created == count_disp, "not equal create-dispose");

    print("done.");
}