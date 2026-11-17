// A class named as a value with no declared type is the class's value, `typeof C`, as `let k: typeof C = C` is:
// its statics, `new` and `instanceof` through the vtable of the class it holds (TSLANG 108). Held by a
// `let`/`var` (module, local, hoisted), returned (`return C`, `return this` in a static, `() => C`), put in an
// object literal field, boxed as `any` and compared. It typed as an instance of C and failed with "failed to
// legalize operation 'ts.ClassRef'". A `const` still names the class itself.
class Base {
    static n = 1;
    static make() { return "base"; }
    static get self() { return this; }
    static me() { return this; }
    who() { return "b"; }
}

class Derived extends Base {
    static n = 2;
    static make() { return "derived"; }
    who() { return "d"; }
}

class Other {}

var mk = Base;
let arrow = () => Derived;

function pick(d: boolean) { if (d) return Derived; return Base; }
function local() { let k = Derived; var v = Base; if (true) { var w = Derived; } return k.n + v.n + w.n; }

function main() {
    assert(mk.n == 1 && mk.make() == "base");
    assert(Base.self.n == 1 && Base.me().make() == "base");
    const o = new mk();
    assert(o.who() == "b" && o instanceof Base);
    assert(local() == 5);

    const f = pick(true);
    assert(f.n == 2 && new f().who() == "d");
    assert(arrow().make() == "derived");

    const holder = { cls: Derived };
    assert(holder.cls.make() == "derived" && new holder.cls().who() == "d");

    let a: any = Base;
    let k = Other;
    assert(a === Base && a !== Other && typeof a == "function");
    assert(k === Other && k !== Base);
    print("done.");
}
