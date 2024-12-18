// @strict-null false
class BazClass {}
function testBoolCasts() {
    print("testBoolCast");
    function boolDie() {
        assert(false, "bool casts");
    }
    let x = "Xy" + "Z";

    if (x) {
    } else {
        boolDie();
    }

    // TODO: finish it
    /*
    if ("") {
        boolDie()
    }
*/

    let v = new BazClass();
    if (v) {
    } else {
        boolDie();
    }
    if (!v) {
        boolDie();
    }
    v = null;
    if (v) {
        boolDie();
    }
    if (!v) {
    } else {
        boolDie();
    }
}

function main() {
    testBoolCasts();
    print("done.");
}
