// `a + "-" + b + "!"` builds one string for the whole chain, not one per `+`. What it gives is
// unchanged: operands are evaluated from the left, `+`s before the first string still add
// (`1 + 2 + "x"` is "3x"), and an operand that is not a string, number or boolean (any, a union,
// an object) goes through the `+` it always did.

class Named {
    constructor(public name: string) {}
    toString() {
        return "<" + this.name + ">";
    }
}

class Holder {
    constructor(public text: string) {}
}

let order = "";
function seen(tag: string): string {
    order = order + tag;
    return tag;
}

function count(n: int): int {
    order = order + n;
    return n;
}

function make(a: string, b: string): string {
    return a + "-" + b + "!";
}

function main() {
    const a = "left";
    let b = "right";
    b = b + "";

    assert(make(a, b) == "left-right!", "variables and constants");
    assert("a" + b + "c" + "d" == "arightcd", "constants after a variable merge");
    assert(b + 1 + 2 == "right12", "numbers after a string are converted one by one");

    let one = 1;
    let two = 2;
    assert(one + two + b == "3right", "numbers before a string add");
    assert(one + two + b + one + two == "3right12", "add, then concatenate");
    assert(b + true + false == "righttruefalse", "booleans");
    assert(b + 0.5 + 1.25 == "right0.51.25", "numbers");
    assert(b + (one + two) + b == "right3right", "parenthesized sum");

    // evaluated from the left, once each
    order = "";
    const s = seen("x") + count(1) + seen("y") + count(2);
    assert(s == "x1y2", "value");
    assert(order == "x1y2", "evaluation order");

    // operands that end the run
    const anyValue: any = 5;
    assert(b + anyValue + "!" == "right5!", "any in the chain");
    const named = new Named("n");
    assert("[" + named + "]" + b == "[<n>]right", "object with toString in the chain");
    let maybe: string | null = null;
    assert("a" + maybe + b + "c" == "anullrightc", "union in the chain");
    maybe = "m";
    assert("a" + maybe + b + "c" == "amrightc", "union holding a string");

    // a chain's string goes wherever a built one does
    const holder = new Holder(a + ":" + b + ";");
    assert(holder.text == "left:right;", "kept by a constructor");
    const list: string[] = [];
    list.push(b + "/" + a + "/" + one);
    assert(list[0] == "right/left/1", "pushed");
    let grown = "";
    grown += a + "," + b + ",";
    assert(grown == "left,right,", "+= of a chain");
    assert(`${a + "." + b + "."}` == "left.right.", "chain in a template");

    // many times over, so that a leak or a double free shows under rc and own
    let total = 0;
    for (let i = 0; i < 2000; i++) {
        const t = "k" + i + ":" + b + "/" + (i * 2) + "!";
        total += t.length;
        const h = new Holder(t + "-" + i + "-");
        total += h.text.length;
    }
    assert(total > 0, "loop");

    print("done.");
}
