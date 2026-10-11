// A `default` clause without a `break` never fell through into the clauses after it: its body
// jumped straight out of the switch, so `default: r += "q"; case 6: r += "x"` gave "q".

function defaultFirst(x: number) {
    let r = "";
    switch (x) {
        default:
            r += "q";
        // fallthrough
        case 6:
        case 7:
            r += "x";
        // fallthrough
        case 8:
            r += "y";
            break;
        case 10:
            r += "z";
            break;
    }

    return r;
}

function defaultInMiddle(x: number) {
    let r = "";
    switch (x) {
        case 6:
            r += "x";
        // fallthrough
        default:
            r += "q";
        // fallthrough
        case 8:
            r += "y";
            break;
        case 9:
            r += "w";
    }

    return r;
}

function defaultLast(x: number) {
    let r = "";
    switch (x) {
        case 1:
            r += "a";
        // fallthrough
        default:
            r += "d";
    }

    return r + ".";
}

function defaultWithBreak(x: number) {
    let r = "";
    switch (x) {
        default:
            r += "q";
            break;
        case 1:
            r += "a";
    }

    return r;
}

function main() {
    assert(defaultFirst(100) == "qxy", "defaultFirst(100)");
    assert(defaultFirst(6) == "xy", "defaultFirst(6)");
    assert(defaultFirst(7) == "xy", "defaultFirst(7)");
    assert(defaultFirst(8) == "y", "defaultFirst(8)");
    assert(defaultFirst(10) == "z", "defaultFirst(10)");

    assert(defaultInMiddle(100) == "qy", "defaultInMiddle(100)");
    assert(defaultInMiddle(6) == "xqy", "defaultInMiddle(6)");
    assert(defaultInMiddle(8) == "y", "defaultInMiddle(8)");
    assert(defaultInMiddle(9) == "w", "defaultInMiddle(9)");

    assert(defaultLast(1) == "ad.", "defaultLast(1)");
    assert(defaultLast(2) == "d.", "defaultLast(2)");

    assert(defaultWithBreak(5) == "q", "defaultWithBreak(5)");
    assert(defaultWithBreak(1) == "a", "defaultWithBreak(1)");

    print("done.");
}
