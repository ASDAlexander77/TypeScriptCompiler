let glb1 = 0;
let dispose_called = false;

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

function try2() {
    let i = 10;
    try {

        using file = new TempFile(".some_temp_file");

        glb1++;
        print("Try");
        if (i > 0)
            throw 1.0;
        print("cont");
    }
    finally {
        glb1++;
        print("Finally");
    }
}

function main2() {
    glb1 = 0;
    try {
        try2();
    }
    catch
    {
        glb1++;
        print("caught");
    }

    assert(glb1 == 3);
    assert(dispose_called);
}


function main() {
    main2();
    print("done.");
}