// A class read by a key known only at run time (a string, or a value of type any): what it reads is the
// member of the object's actual class (its member table, CLASS_VTABLE_MEMBERS_SLOT), as JavaScript reads it -
// a D held as a C has D's fields. A method read so is bound to the object.
class C {
    a = 1;
    o?: number;
    m(): number {
        return this.a + 10;
    }
}

class D extends C {
    d = "dee";
    override m(): number {
        return 99;
    }
}

class N {
    "0" = 7;
}

class N1 {
    "1" = 9;
}

class WithAccessor {
    get v(): number {
        return 1;
    }
}

function key(s: string) {
    return s;
}

function main() {
    const c: C = new D();

    // a string key: any, the actual class's member
    assert(<string>c[key("d")] == "dee");
    assert(<number>c[key("a")] == 1);
    assert(typeof c[key("nope")] == "undefined");

    // an optional field not set is not there
    assert(!(key("o") in c));
    assert(typeof c[key("o")] == "undefined");
    c.o = 5;
    assert(key("o") in c);
    assert(<number>c[key("o")] == 5);

    // `in` sees the actual class too
    assert(key("d") in c);

    // a method: bound to the object, the actual class's override
    const f = <() => number>c[key("m")];
    assert(f() == 99);

    // through any
    const x: any = c;
    assert(<string>x[key("d")] == "dee");

    // the base's storage a subclass begins with, and an accessor's functions, are no keys
    assert(!(key("C") in x));
    assert(typeof x[key("C")] == "undefined");
    const h: any = new WithAccessor();
    assert(key("v") in h);
    assert(!(key("get_v") in h));

    // a number key is its text
    const n: any = new N();
    assert(<number>n[0] == 7);

    // ...a run-time number too, on the class's own type
    let i = 0;
    i++;
    const n1 = new N1();
    assert(<number>n1[i] == 9);

    print("done.");
}
