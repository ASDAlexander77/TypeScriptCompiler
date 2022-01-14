// TODO: bug.  because x class and class in foo are different types, static fields are not copied
function foo(x = class { static prop: string }): string {
    return undefined;
}

function main() {
    foo(class { static prop = "hello" }).length;

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