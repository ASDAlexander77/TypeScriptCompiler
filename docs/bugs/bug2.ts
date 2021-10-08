function call_func(f: (o: { firstName: string }) => void) {
    let user2 = {
        firstName: "my2",
    };

    f(user2);
}

function main() {
    const user = {
        firstName: "my",
        sayHi() {
            print(`in`);
            print(`Hello ${this.firstName}`);
        },
    };

    user.sayHi();
    /*

    //const hi = user.sayHi;
    //hi();

    let hi2 = user.sayHi;
    //hi2();

    //call_func(() => {hi2();});

    call_func(hi2);
*/

    call_func(user.sayHi);

    print("done.");
}
