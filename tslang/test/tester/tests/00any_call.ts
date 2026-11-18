// A function boxed in an `any` is called through it (user ruling 11-18): each argument is cast to the
// parameter's type as any value is, the result is boxed, and what holds no function is a TypeError
class C {
    x = 7;
}

function twice(n: number) {
    return n * 2;
}

function greet(s: string, punct?: string): string {
    return "hi " + s + (punct ?? ".");
}

function main() {
    const a: any = twice;
    assert(<number>a(21) == 42, "plain function");

    const g: any = greet;
    assert(<string>g("bob") == "hi bob.", "optional parameter left out");
    assert(<string>g("bob", "!") == "hi bob!", "optional parameter given");

    const captured = 10;
    const add: any = (n: number) => n + captured;
    assert(<number>add(5) == 15, "closure");

    let calls = 0;
    const v: any = () => {
        calls++;
    };
    v();
    assert(calls == 1, "void function");

    const take: any = (c: C) => c.x;
    assert(<number>take(new C()) == 7, "class argument");

    let classArgumentRefused = false;
    try {
        take(5);
    } catch (e) {
        classArgumentRefused = true;
    }

    assert(classArgumentRefused, "a number is not a C");

    let fewArgumentsRefused = false;
    try {
        a();
    } catch (e) {
        fewArgumentsRefused = true;
    }

    assert(fewArgumentsRefused, "too few arguments");

    const notFunction: any = 5;
    let notFunctionRefused = false;
    try {
        notFunction(1);
    } catch (e) {
        notFunctionRefused = true;
    }

    assert(notFunctionRefused, "not a function");

    // a cast from any to a function type checks what the box holds
    let castRefused = false;
    try {
        const f = <(n: number) => number>notFunction;
    } catch (e) {
        castRefused = true;
    }

    assert(castRefused, "cast of a number to a function");

    const nothing: any = null;
    const maybe = <((n: number) => number) | null>nothing;
    assert(maybe === null, "null to a function or null");

    print("done.");
}
