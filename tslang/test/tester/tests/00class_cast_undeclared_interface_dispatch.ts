// A class value cast to an interface its class does not declare (`const n: Named = k`): the interface's
// vtable is built from the static type, once, and serves an object of any subclass. A method it names
// has to be the object's own - the override - so its slot calls through the object's vtable. It named
// the static type's method directly: an override was never called (Base's for a Derived), and an
// abstract method was a symbol with no body, which failed to link (JIT: "Symbols not found").

interface Named {
    title(): string;
}

interface Labeled {
    id: number;
    label(prefix: string, n: number): string;
}

abstract class Kind {
    abstract title(): string;
}

class Concrete extends Kind {
    title() {
        return "concrete";
    }
}

class Base {
    id: number = 1;

    title() {
        return "base";
    }

    label(prefix: string, n: number) {
        return prefix + "base" + n;
    }

    plain() {
        return "plain";
    }
}

class Derived extends Base {
    title() {
        return "derived";
    }

    label(prefix: string, n: number) {
        return prefix + "derived" + n;
    }
}

class Leaf extends Derived {
    other: number = 3;
}

interface Plain {
    plain(): string;
}

function titleOf(n: Named) {
    return n.title();
}

function main() {
    const k: Kind = new Concrete();
    const fromAbstract: Named = k;
    assert(fromAbstract.title() == "concrete", "an abstract method: the subclass's");

    const b: Base = new Derived();
    const fromBase: Named = b;
    assert(fromBase.title() == "derived", "an override, through the base's type");
    assert(titleOf(b) == "derived", "an override, as an argument");

    const labeled: Labeled = b;
    assert(labeled.label("<", 2) == "<derived2", "an override with arguments and a string result");
    assert(labeled.id == 1, "a field next to it");

    const leaf: Base = new Leaf();
    const fromLeaf: Labeled = leaf;
    assert(fromLeaf.label("", 7) == "derived7", "an inherited override");
    assert(fromLeaf.id == 1, "an inherited field");

    const own: Base = new Base();
    const fromOwn: Named = own;
    assert(fromOwn.title() == "base", "the static type's own method");

    const plain: Plain = b;
    assert(plain.plain() == "plain", "a method no subclass overrides");

    print("done.");
}
