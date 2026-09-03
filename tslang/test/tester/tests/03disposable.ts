let dispose_called = false;

class Res {
    [Symbol.dispose]() {
        dispose_called = true;
        print("disposed");
    }
}

// no try/catch here at all: the exception must unwind straight through this using-scope,
// and dispose must still run on the way out
function inner() {
    using r = new Res();
    print("in inner");
    throw 1;
}

function main() {
    try {
        inner();
    }
    catch (e: TypeOf<1>) {
        print("caught");
    }

    assert(dispose_called, "dispose is not called when unwinding through a using with no enclosing try");

    print("done.");
}
