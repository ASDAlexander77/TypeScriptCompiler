// A field of a union of object shapes or classes is read from the member the value holds (TSLANG 112):
// the members' fields may sit at other positions or be of other types.
type Sq = { size: number; kind: "sq" };
type Ci = { r: number; name: string; kind: "ci" };
function kindOf(s: Sq | Ci) { return s.kind; }
function measure(s: Sq | Ci) {
    if (s.kind === "sq") return s.size * s.size;
    return s.r * 3;
}

type First1 = { kind: "a"; n: number };
type First2 = { kind: "b"; text: string; m: number };
function firstKind(v: First1 | First2) { return v.kind; }

function label(options: { label: string; alias: string } | { label: object; alias?: string }) {
    if (typeof options.label === "string") return options.alias ?? "string";
    return "object";
}

class A { x = 1; s = "a" + 1; }
class B { s = "b" + 2; }
class C extends B { y = 3; s2 = "c"; }
function sOf(v: A | B) { return v.s; }
function sOf3(v: A | B | C) { return v.s; }

function main() {
    assert(kindOf({ kind: "sq", size: 2 }) == "sq", "kind of a square");
    assert(kindOf({ kind: "ci", r: 5, name: "c" + 1 }) == "ci", "kind of a circle");
    assert(measure({ kind: "sq", size: 2 }) == 4, "square");
    assert(measure({ kind: "ci", r: 5, name: "c" + 1 }) == 15, "circle");

    assert(firstKind({ kind: "a", n: 1 }) == "a", "first field a");
    assert(firstKind({ kind: "b", text: "t", m: 2 }) == "b", "first field b");

    assert(label({ label: "x", alias: "y" }) == "y", "label string");
    assert(label({ label: <object>{}, alias: "z" }) == "object", "label object with alias");

    assert(sOf(new A()) == "a1", "class A");
    assert(sOf(new B()) == "b2", "class B");
    assert(sOf3(new C()) == "b2", "subclass C");

    for (let i = 0; i < 3; i++) {
        const xs: (Sq | Ci)[] = [{ kind: "sq", size: i }, { kind: "ci", r: i, name: "n" + i }];
        assert(kindOf(xs[0]) == "sq" && kindOf(xs[1]) == "ci", "in a loop");
    }

    print("done.");
}
