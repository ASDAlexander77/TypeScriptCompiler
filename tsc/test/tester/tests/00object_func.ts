function main() {
    let user = {
        firstName: "John",
        sayHi() {
            print(`Hello, ${this.firstName}!`);
        },
    };

    user.sayHi();

    print("done.");
}
