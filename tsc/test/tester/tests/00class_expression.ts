function foo(x = class { static prop: string }): string {
    return x.prop;
}

function foo2(x = class { static prop: string; static func(); }) {
    x.func();
}

function main() {
    assert(foo(class { static prop = "hello" }).length == 5);

    foo2(class {
        static prop = "asdasd";
        static func() {
            print("Hello World 2", this.prop);
        }
    });

    main2();

    print("done.");
}

function main2() {
    const a = class { static prop = "hello" };

    function f(p: typeof a) {
        print(p.prop);
    }

    f(a);
}