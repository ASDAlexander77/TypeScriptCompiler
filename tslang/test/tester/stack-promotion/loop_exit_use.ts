// A use on the way out of a loop - a return, a break out of a search - comes after the last
// iteration: the one stack slot is not reused after it, so the object goes on the stack.
class Holder {
    s: string;
    constructor(s: string) {
        this.s = s;
    }
}

function findReturn(n: number) {
    for (let i = 0; i < n; i++) {
        const p = new Holder("v" + i);
        if (i == 2) {
            return p.s;
        }
    }
    return "";
}

function findBreak(n: number) {
    let found = "";
    for (let i = 0; i < n; i++) {
        const p = new Holder("w" + i);
        if (i == 3) {
            found = p.s;
            break;
        }
    }
    return found;
}

function main() {
    assert(findReturn(10) == "v2");
    assert(findReturn(1) == "");
    assert(findBreak(10) == "w3");
    assert(findBreak(2) == "");
    print("done.");
}
