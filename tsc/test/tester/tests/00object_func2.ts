function call_func_1(f: () => void) {
    f();
}

function call_func(f: (o: object) => void, user: { firstName: string }) {
    f(user);
}

function call_func2(f: (o: unknown) => void, user: { firstName: string }) {
    f(user);
}

function main() {
    const user = {
        firstName: "World",
        sayHi() {
            print(`Hello ${this.firstName}`);
        },
    };

    user.sayHi();

    const hi = user.sayHi;
    hi();

    let hi2 = user.sayHi;
    hi2();

    call_func_1(() => {
        hi2();
    });

    call_func(user.sayHi, user);
    // We do not allow it
    //call_func2(user.sayHi, user);

    print("done.");
}
