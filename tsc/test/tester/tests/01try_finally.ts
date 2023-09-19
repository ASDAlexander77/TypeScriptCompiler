// inline test

function mayThrow(i: number) {
    if (i > 10.0)
        throw 1.0;
}

function main4() {

    try {
        print("try");
        mayThrow(100);
    }
    finally {
        print("finally");
    }
}

function main() {
    try {
        main4();
    }
    catch (e: number) {
        print("catch outer");
    }

    print("done.");
}