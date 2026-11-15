// TSLANG 85: a global whose initializer concatenates three or more strings. The inner sum is a
// temporary the initializer reads and gives back itself; -mm=own took its reference for a second
// one the global takes ("the initializer of global 'M.EXPORTS' takes a second reference").
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
    print("done.");
}
