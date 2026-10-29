// Generic methods in an interface: each is one vtable slot erased to `any` for its type parameters.
// A class fills it with its generic method instantiated with `any`; a call boxes the arguments and
// unboxes the result to the type its type arguments - given or inferred - make it.
interface Mapper {
    map<T>(x: T): T;
    tag<T>(x: T, label: string): string;
    choose<A>(a: A, b: A, first: boolean): A;
}

class Box {
    constructor(public n: number) {}
}

class Id implements Mapper {
    map<T>(x: T): T {
        return x;
    }

    tag<T>(x: T, label: string): string {
        return "id:" + label;
    }

    choose<A>(a: A, b: A, first: boolean): A {
        if (first) return a;
        return b;
    }
}

class Logged implements Mapper {
    count = 0;

    map<T>(x: T): T {
        this.count++;
        return x;
    }

    tag<T>(x: T, label: string): string {
        this.count++;
        return "logged:" + label;
    }

    choose<A>(a: A, b: A, first: boolean): A {
        this.count++;
        if (first) return a;
        return b;
    }
}

// inherits its generic methods from Id
class Sub extends Id {
    extra() {
        return 1;
    }
}

function use(m: Mapper) {
    return m.map<number>(40) + 2;
}

function main() {
    const m: Mapper = new Id();
    const n: number = m.map<number>(5);
    assert(n + 1 == 6, "explicit");

    const s = m.map("ab");
    assert(s.length == 2, "inferred string");

    const b = m.map(new Box(7));
    assert(b.n == 7, "class object");

    assert(m.tag(1, "x") == "id:x", "non-generic result");

    const c = m.choose<string>("zero", "one", false);
    assert(c == "one", "two parameters of one type");

    const l = new Logged();
    const lm: Mapper = l;
    assert(lm.map(3) == 3 && lm.tag("q", "y") == "logged:y", "another implementer");
    assert(l.count == 2, "its own state");

    assert(use(new Id()) == 42 && use(l) == 42 && use(new Sub()) == 42, "dispatched");

    // a call on the class itself is instantiated for its type arguments, as before
    assert(new Id().map<number>(8) == 8, "direct");

    print("done.");
}
