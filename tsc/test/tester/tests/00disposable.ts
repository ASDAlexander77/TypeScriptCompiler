let called = false;

function main() {
    doSomeWork1();

    assert(called, "Disposable is not called");

    called = false;

    {
        using file = new TempFile(".some_temp_file");
    }    

    assert(called, "Disposable is not called");

    print("done.");
}

interface Disposable {
    [Symbol.dispose](): void;
}

function doSomeWork1() {
    const file = new TempFile(".some_temp_file");
    try {
        // ...
    }
    finally {
        file[Symbol.dispose]();
    }
}

class TempFile implements Disposable {
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
        called = true;
        print("dispose");
    }
}