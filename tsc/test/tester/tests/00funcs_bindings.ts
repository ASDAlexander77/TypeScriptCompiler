function objectBindingPattern({ foo }: { foo: number }) {
    print(foo);
    assert(foo == 10.0);

}

function arrayBindingPattern([foo]: number[]) {
    print(foo);
    assert(foo == 1.0);
}

function main() {
    objectBindingPattern({ val: 10.0 });
    arrayBindingPattern([1.0]);
    print("done.");
}