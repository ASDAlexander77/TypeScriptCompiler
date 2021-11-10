function test_const() {
    const user = {
        firstName: "John",
        sayHi() {
            print(`Hello, ${this.firstName}!`);
        },
    };

    user.sayHi();
}

function test_let() {
    let user = {
        firstName: "John",
        sayHi() {
            print(`Hello, ${this.firstName}!`);
        },
    };

    user.sayHi();
}

function test_nest_capture_of_this() {
    let deck = {
        val: 1,
        funcWithCapture: function () {
            return () => {
                return this.val;
            };
        },
    };

    let funcInst = deck.funcWithCapture();
    // BUG: double call causes crash (due to Trampoline)
    //print(funcInst());
    assert(funcInst() == 1);
}

function main() {
    test_const();
    test_let();
    test_nest_capture_of_this();
    print("done.");
}
