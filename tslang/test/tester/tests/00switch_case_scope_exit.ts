// The cases of a `switch` share a scope of their own: a `using` in a case is disposed where the
// switch ends - by a `break`, by the last case falling off its end, or by no case matching - and
// a jump out of a case disposes it on the way.
//
// The cases used to be generated in the enclosing scope's lists, while their names lived only in
// the switch's symbol table scope: any `using` in a case crashed the compiler when the enclosing
// scope ended and looked the name up. A `break` written inside an `if` in a case had no target to
// lower to and crashed the compiler too, and on its way out it released the locals of every scope
// around the switch.

let trail = "";

class Res {
    constructor(public name: string) {}

    [Symbol.dispose]() {
        trail += this.name + ";";
    }
}

function caseWithBreak(i: number) {
    switch (i) {
        case 0:
            using a = new Res("a");
            trail += "body0;";
            break;
        case 1:
            trail += "body1;";
            break;
    }

    trail += "after;";
}

function fallThrough(i: number) {
    switch (i) {
        case 0:
            using a = new Res("a");
            trail += "body0;";
        case 1:
            trail += "body1;";
            break;
        default:
            trail += "default;";
    }

    trail += "after;";
}

function bracedCases(i: number) {
    switch (i) {
        case 0: {
            using a = new Res("a");
            trail += "body0;";
        }
        case 1: {
            using b = new Res("b");
            trail += "body1;";
        }
    }

    trail += "after;";
}

function breakFromIf(i: number) {
    using f = new Res("fn");
    switch (i) {
        case 0:
            using a = new Res("a");
            if (a.name == "a") break;
            trail += "unreached;";
        case 1:
            trail += "body1;";
    }

    trail += "after;";
}

function inLoop() {
    for (let j = 0; j < 4; j++) {
        using l = new Res("loop" + j);
        switch (j) {
            case 0:
                using c = new Res("case" + j);
                continue;
            case 1:
                using d = new Res("case" + j);
                break;
            default:
                using e = new Res("default" + j);
                if (j == 2) break;
                trail += "default;";
        }

        trail += "end" + j + ";";
    }

    trail += "after;";
}

class Shape {
    constructor(public kind: string) {}
}

function onProperty(s: Shape) {
    switch (s.kind) {
        case "circle":
            trail += "circle;";
            break;
        case "square":
            using q = new Res("square");
            trail += "body;";
            break;
    }

    trail += "after;";
}

// strings: under rc a case's locals are released by the switch, and a `break` from an `if`
// no longer releases the locals around the switch
function stringsInCases(i: number, k: string) {
    let outer = "o" + k;
    switch (i) {
        case 0:
            let s = "zero" + k;
            if (s.length > 4) break;
            outer += s;
        case 1:
            let t = "one" + k;
            outer += t;
            break;
        default:
            let u = "def" + k;
            outer += u;
    }

    return outer;
}

function stringsInLoop() {
    let acc = "";
    for (let j = 0; j < 4; j++) {
        let v = "v" + j;
        switch (j) {
            case 0:
                let w = v + "w";
                acc += w;
                continue;
            case 1:
                acc += v;
                break;
            default:
                let x = v + "x";
                if (j == 2) {
                    acc += x;
                    break;
                }

                acc += "!";
        }

        acc += ";";
    }

    return acc;
}

function check(f: () => void, expected: string, what: string) {
    trail = "";
    f();
    print(what + ": " + trail);
    assert(trail == expected, what);
}

function main() {
    check(() => caseWithBreak(0), "body0;a;after;", "using + break");
    check(() => caseWithBreak(1), "body1;after;", "using in a case not taken");
    check(() => caseWithBreak(5), "after;", "no case matched");
    check(() => fallThrough(0), "body0;body1;a;after;", "falling through");
    check(() => fallThrough(7), "default;after;", "default");
    check(() => bracedCases(0), "body0;a;body1;b;after;", "braced cases falling through");
    check(() => breakFromIf(0), "a;after;fn;", "break from an if in a case");
    check(() => breakFromIf(1), "body1;after;fn;", "break from an if, case not taken");
    check(inLoop, "case0;loop0;case1;end1;loop1;default2;end2;loop2;default;default3;end3;loop3;after;",
          "switch in a loop body with a using");
    check(() => onProperty(new Shape("square")), "body;square;after;", "switch on a property");
    check(() => onProperty(new Shape("circle")), "circle;after;", "switch on a property, other case");

    assert(stringsInCases(0, "") == "ozeroone", "strings, fall through");
    assert(stringsInCases(0, "long") == "olong", "strings, break from an if");
    assert(stringsInCases(1, "a") == "oaonea", "strings, case 1");
    assert(stringsInCases(9, "b") == "obdefb", "strings, default");
    assert(stringsInLoop() == "v0wv1;v2x;!;", "strings in a loop");

    print("done.");
}
