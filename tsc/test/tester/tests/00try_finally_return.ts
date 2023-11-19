let called = false;

function may_throw(a = 0) {
    if (a > 100) throw 1;
}

function func1() {
    try {
        print("In try");
        may_throw(10);
        return;
    }
    finally {
        print("finally");
        called = true;
    }

    called = false;
    print("end");
}

function func2(a = 0) {
    try {
        print("In try");
        may_throw(1);
        if (a > 20)
            return;
    }
    catch (e: TypeOf<1>) {
        print("catch");
        throw e;
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
    called = false;
    func2(100);
    assert(called, "finally is not called");
    print("done.");
}