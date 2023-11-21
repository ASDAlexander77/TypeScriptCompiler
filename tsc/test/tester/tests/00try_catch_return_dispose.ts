let called = false;
let dispose_called = false;

function may_throw(a = 0) {
    if (a > 100) throw 1;
}

class TempFile {
    #path: string;
    #handle: number;
    constructor(path: string) {
        this.#path = path;
        this.#handle = 1;
    }
    // other methods
    [Symbol.dispose]() {
        // Close the file and delete it.
        this.#handle = 0;
        dispose_called = true;
        print("dispose");
    }
}

function func1() {

    try {
        using file = new TempFile(".some_temp_file");
        print("In try");
        may_throw(1000);
    }
    catch (e: TypeOf<1>) {
        print("catch");
        return;
    }
    finally {
        print("finally");
        called = true;
    }

    called = false;
    print("end");
}

function func2() {

    try {
        using file = new TempFile(".some_temp_file");
        print("In try");
        may_throw(1);
    }
    catch (e: TypeOf<1>) {
        print("catch");
        return;
    }
    finally {
        print("finally");
        called = true;
    }

    called = false;
    print("end");
}

function main() {
    func1();
    assert(called, "finally is not called");
    assert(dispose_called, "dispose is not called");
    called = false;
    dispose_called = false;
    func2();
    assert(called, "finally is not called");
    assert(dispose_called, "dispose is not called");
    print("done.");
}