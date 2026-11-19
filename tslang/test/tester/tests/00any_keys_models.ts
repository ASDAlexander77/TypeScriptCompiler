// `x.name`, `x[k]`, `x[k] = v` and `k in x` of a value of type `any` under every memory model, -mm=own
// too: a key whose value copies (a number, a boolean, a string) is read and written; under -mm=own a key
// holding an object, a write through an interface, and an array grown through `any` are TypeErrors.
class C {
    a = 1;
    s = "x";
    b = true;

    get g() { return this.a * 10; }

    m() { return 1; }
}

interface I {
    n: number;
    o?: string;
}

function main() {
    const x: any = new C();
    assert(x.a == 1 && x.s == "x" && x.b && x.g == 10, "read by name");
    assert("a" in x && "m" in x && !("zz" in x), "in");

    x.a = 5;
    x["s"] = "y" + x.a;
    x.b = false;
    assert(x.a == 5 && x.s == "y5" && !x.b && x.g == 50, "written");

    x.a += 2;
    x.a++;
    assert(x.a == 8, "compound and increment");

    let refused = 0;
    try { x.g = 1; } catch (e) { refused++; }
    try { x.zz = 1; } catch (e) { refused++; }
    try { x.a = "a string"; } catch (e) { refused++; }
    assert(refused == 3 && x.a == 8, "what is refused");

    for (let k = 0; k < 50; k++) {
        x.s = "s" + k;
    }

    assert(x.s == "s49", "in a loop");

    const y: any = <I>{ n: 1, o: "a" };
    assert(y.n == 1 && y.o == "a" && "o" in y, "an interface");

    const z: any = [1, 2, 3];
    z[1] = 20;
    assert(z.length == 3 && z[1] == 20 && z[0] == 1 && typeof z[5] == "undefined", "an array");

    const strs: any = ["a", "b"];
    strs[0] = "c" + z.length;
    assert(strs[0] == "c3", "a string array");

    const s: any = "str";
    assert(s.length == 3 && s[1] == "t", "a string");

    print("done.");
}
