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

function main_func_in_object2(a?: number) {
    const s = {
        f() {
            assert(a == 11);
            print(a);
        },
    };

    s.f();
}

function main_func_in_object3(a: number | undefined) {
    const s = {
        f() {
            assert(a == 11);
            print(a);
        },
    };

    s.f();
}

function main_func_in_object4(a?: number | undefined) {
    const s = {
        f() {
            assert(a == 11);
            print(a);
        },
    };

    s.f();
}

namespace exceptions {

    function lambda(k: number) {
        function inner() {
            print("inner inside", k);
        }
        print("inner");
        inner()
        print("exit inner");
    }

}


function lambda(k: number) {
    function inner() {
        print("inner inside", k);
    }
    print("inner");
    inner()
    print("exit inner");
}

function main_namespace_inner() {
    lambda(12);
    exceptions.lambda(12);
}

function main() {
    main_func();
    main_func_in_object();
    main_func_in_object2(11);
    main_func_in_object3(11);
    main_func_in_object4(11);

    main_namespace_inner();

    print("done.");
}
