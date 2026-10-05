// a `for` body sees the variable its condition narrows, as a `while` body does: `typeof v === "string"` made
// `v.length` "Can't resolve property 'length' of type number". The incrementor still assigns the variable as
// it is declared.
class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

class A {
    constructor(public a: number) {}
    onlyA() {
        return this.a;
    }
}

class B {
    constructor(public b: number) {}
}

type Sq = { kind: "sq"; side: number };
type Ci = { kind: "ci"; r: number };

function isStr(v: string | number): v is string {
    return typeof v === "string";
}

function typeofIncrementor(v: string | number) {
    let n = 0;
    for (; typeof v === "string"; v = 1) {
        n += v.length;
    }

    return n;
}

function typeofBodyWrite(v: string | number) {
    let n = 0;
    for (; typeof v === "string"; ) {
        n += v.length;
        v = 1;
    }

    return n;
}

function typeGuard(v: string | number) {
    let n = 0;
    for (; isStr(v); v = 1) {
        n += v.length;
    }

    return n;
}

function instanceOf(x: A | B) {
    let n = 0;
    for (; x instanceof A; x = new B(1)) {
        n += x.onlyA();
    }

    return n;
}

function discriminant(s: Sq | Ci) {
    let n = 0;
    for (; s.kind === "sq"; ) {
        n += s.side;
        s = { kind: "ci", r: 1 };
    }

    return n;
}

function listSum(head: Node | undefined) {
    let sum = 0;
    for (let node = head; node; node = node.next) {
        const n: Node = node;
        sum += n.value;
    }

    return sum;
}

function listSkip(head: Node | undefined) {
    let sum = 0;
    for (let node = head; node; node = node.next) {
        if (node.value == 2) {
            continue;
        }

        sum += node.value;
    }

    return sum;
}

function nested(head: Node | undefined) {
    let sum = 0;
    for (let a = head; a; a = a.next) {
        for (let b = head; b; b = b.next) {
            sum += a.value * b.value;
        }
    }

    return sum;
}

function staticFalse() {
    const a = [1, 2];
    let n = 0;
    for (; typeof a === "string"; ) {
        n += a.length;
    }

    return n;
}

function main() {
    assert(typeofIncrementor("abcd") == 4, "typeof, incrementor writes");
    assert(typeofIncrementor(7) == 0, "typeof, not entered");
    assert(typeofBodyWrite("abcde") == 5, "typeof, body writes");
    assert(typeGuard("abc") == 3, "type guard");
    assert(instanceOf(new A(5)) == 5, "instanceof");
    assert(discriminant({ kind: "sq", side: 4 }) == 4, "discriminant, body writes the other member");

    let head: Node | undefined = new Node(1, new Node(2, new Node(3, undefined)));
    assert(listSum(head) == 6, "optional list walk");
    assert(listSum(undefined) == 0, "empty list");
    assert(listSkip(head) == 4, "continue runs the incrementor");
    assert(nested(head) == 36, "nested loops");
    assert(staticFalse() == 0, "a condition known to be false");

    for (let i = 0; i < 1; i++) var x = 5;
    assert(x == 5, "var in an unbraced body");

    print("done.");
}
