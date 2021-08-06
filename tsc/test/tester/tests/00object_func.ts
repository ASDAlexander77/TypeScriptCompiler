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

function main() {
    test_const();
    test_let();
    print("done.");
}
