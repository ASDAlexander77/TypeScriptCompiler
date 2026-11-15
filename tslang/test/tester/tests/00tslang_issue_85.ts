// TSLANG 85: a global whose initializer concatenates three or more strings. The inner sum is a
// temporary the initializer reads and gives back itself; -mm=own took its reference for a second
// one the global takes ("the initializer of global 'M.EXPORTS' takes a second reference"). So it
// did for a sum given to a call, even of two strings: `new Holder("x" + "y")`, and a chain returning
// `this` on a new object was refused as a borrow: `new Bag().add("m")`.
namespace M {
    const EXPORTS = "Raygen,Miss," +
        "Hit," + "Group";

    function show(names: string) {
        print(names);
        return names.length;
    }

    export function run() {
        return show(EXPORTS);
    }
}

const FOUR = "a" + "b" + "c" + "d";
const MIXED = "n=" + 1 + "," + 2;
const FROM_GLOBALS = FOUR + "-" + MIXED + "-" + FOUR;

const TEMPLATE = `${FOUR}!` + "?" + MIXED;

class H {
    text: string = "t" + "u";
}

const G = new H();
const FROM_FIELD = G.text + "x" + "y";
const ROWS = ["p" + "q"];
const FROM_ELEMENT = ROWS[0] + "x" + "y";

// given to a constructor that keeps it, and to a function that only reads it
class Holder {
    constructor(public text: string) {}
}

function len(s: string) {
    return s.length;
}

const HELD = new Holder("x" + "y" + "z");
const HELD_TWO = new Holder("p" + "q");
const LENGTH = len("a" + "b" + "c");
const BOTH = new Holder("u" + "v").text + "w" + len("r" + "s");

// a chain of calls returning `this` on a new object: the result is that object, the global's own
class Bag {
    items: string[] = [];

    add(s: string): this {
        this.items.push(s);
        return this;
    }
}

class Counter {
    n = 0;

    bump() {
        this.n++;
        return this;
    }
}

const BAG = new Bag().add("m").add("n" + "o");
const ONE = new Bag().add("p");
const COUNTER = new Counter().bump().bump();
const COUNT = new Bag().add("a").add("b" + "c").items.length;

function main() {
    for (let i = 0; i < 1000; i++) {
        assert(FOUR == "abcd", "four");
    }

    assert(M.run() == 21, "namespace");
    assert(MIXED == "n=1,2", "mixed");
    assert(FROM_GLOBALS == "abcd-n=1,2-abcd", "from globals");
    assert(TEMPLATE == "abcd!?n=1,2", "template");
    assert(FROM_FIELD == "tuxy" && G.text == "tu", "field");
    assert(FROM_ELEMENT == "pqxy" && ROWS[0] == "pq", "element");
    assert(HELD.text == "xyz" && HELD_TWO.text == "pq", "held");
    assert(LENGTH == 3 && BOTH == "uvw2", "call");
    assert(BAG.items.length == 2 && BAG.items[0] == "m" && BAG.items[1] == "no", "chain");
    assert(ONE.items[0] == "p" && COUNTER.n == 2 && COUNT == 2, "chains");
    print("done.");
}
