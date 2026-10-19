// -mm=own: an owning local returned on more than one way out moves once on each: no path reaches
// one return from another. And an array a spread builds is moved out when it is returned
// (`return [...r]`), as it is when it is read into a `let`.
class Node {
    constructor(public v: int) {}
}

class Pairs {
    [index: number]: string;

    get(i: number) {
        return "p" + i;
    }

    get length() {
        return 2;
    }
}

function node(x: int): Node {
    let r = new Node(x);
    if (x % 2 == 0) {
        return r;
    }

    r.v += 100;
    return r;
}

function list(x: int): string[] {
    let r: string[] = [];
    if (x == 0) {
        return r;
    }

    for (let i = 0; i < x; i++) {
        r.push("s" + i);
    }

    return r;
}

function spread(p: Pairs): string[] {
    return [...p];
}

function copy(a: string[]): string[] {
    return [...a, ...a];
}

function churn() {
    const junk: Node[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Node(-1));
    }
}

function main() {
    let t = 0;
    const p = new Pairs();
    for (let i = 0; i < 10000; i++) {
        const n = node(i);
        const l = list(i % 4);
        const s = spread(p);
        const c = copy(l);
        churn();
        t += (n.v - i) + l.length + s.length + c.length;
        assert(s[1] == "p1");
    }

    // node: +100 on the odd half; list: 0+1+2+3 per 4; spread: 2; copy: twice list
    assert(t == 5000 * 100 + 2500 * 6 + 10000 * 2 + 2500 * 12);
    print("done.");
}
