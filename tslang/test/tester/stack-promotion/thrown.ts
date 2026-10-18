// A thrown object outlives the function that threw it.
class Oops {
    code: number;
    constructor(code: number) {
        this.code = code;
    }
}

function fail(code: number) {
    const e = new Oops(code);
    throw e;
}

function main() {
    try {
        fail(7);
    } catch (e) {
        print("caught");
    }
    print("done.");
}
