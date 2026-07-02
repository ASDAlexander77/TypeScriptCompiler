let called = false;

function may_throw(a = 0) {
    if (a > 100) throw 1;
}

function func1() {

    try {
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

function main() {
    func1();
    assert(called, "finally is not called");
    print("done.");
}