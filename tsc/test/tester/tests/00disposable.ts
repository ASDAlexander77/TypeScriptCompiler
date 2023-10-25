let called = false;

function main() {
    doSomeWork();

    assert(called, "Disposable is not called");

    print("done.");
}

interface Disposable {
    [Symbol.dispose](): void;
}

function doSomeWork() {
    const file = new TempFile(".some_temp_file");
    try {
        // ...
    }
    finally {
        file[Symbol.dispose]();
        called = true;
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
        print("dispose");
    }
}