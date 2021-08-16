function main_func() {
    let a = 10;

    function f() {
        assert(a == 10);
        print(a);
    }

    f();
}

function main_func_in_object() {
    let a = 10;

    const s = {
        in: 20,
        f() {
            assert(this.in == 20);
            assert(a == 10);
            print(this.in, a);
        },
    };

    s.f();
}

function main() {
    main_func();
    main_func_in_object();
    print("done.");
}
