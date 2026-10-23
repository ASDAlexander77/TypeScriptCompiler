// A `const` holding a call's result (a folded const, released where its block ends) stays alive
// when a `break`, `continue` or `return` leaves its block through a `try` whose `finally` reads
// it, or leaves by a jump of its own. Under rc the release went in front of the jump, and the
// finally then ran on a freed block, or released it a second time in front of its own jump.
// Each read follows allocations of the same size, so a freed block is reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }

    get() {
        return this.x;
    }
}

function mv(x: number) {
    return new V(x);
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(999));
    }
}

function viaContinue() {
    let s: number = 0;
    for (let i = 0; i < 4; i++) {
        const v = mv(i);
        try {
            churn();
            if (i % 2 == 0) {
                continue;
            }
        } finally {
            churn();
            s += v.get();
        }
    }

    return s;
}

function viaBreak() {
    let s: number = 0;
    for (let i = 0; i < 4; i++) {
        const v = mv(i + 10);
        try {
            if (i == 1) {
                break;
            }
        } finally {
            churn();
            s += v.x;
        }
    }

    return s;
}

function viaTwoJumps() {
    let s: number = 0;
    for (let i = 0; i < 6; i++) {
        const v = mv(i);
        try {
            if (i % 3 == 0) {
                continue;
            }

            if (i % 3 == 1) {
                continue;
            }
        } finally {
            churn();
            s += v.get();
        }
    }

    return s;
}

function viaLabelledContinue() {
    let s: number = 0;
    outer: for (let i = 0; i < 3; i++) {
        const v = mv(i + 1);
        for (let j = 0; j < 3; j++) {
            try {
                try {
                    if (j == 1) {
                        continue outer;
                    }
                } finally {
                    churn();
                    s += v.get();
                }
            } finally {
                churn();
                s += v.x * 100;
            }
        }
    }

    return s;
}

function viaReturn(x: number) {
    const v = mv(x);
    try {
        if (x > 0) {
            return x;
        }
    } finally {
        churn();
        assert(v.x == x && v.tag == "v" + x, "read in the finally of a return");
    }

    return -1;
}

function jumpInFinally() {
    let s: number = 0;
    for (let i = 0; i < 4; i++) {
        const v = mv(i);
        try {
            try {
                if (i % 2 == 0) {
                    continue;
                }
            } finally {
                churn();
                if (i == 2) {
                    break;
                }
            }
        } finally {
        }

        s += v.x;
    }

    churn();
    return s;
}

function callInFinally(v: V) {
    churn();
    return v.get();
}

function viaCallInFinally() {
    let s: number = 0;
    for (let i = 0; i < 3; i++) {
        const v = mv(i + 5);
        try {
            if (v.x > 2) {
                continue;
            }
        } finally {
            s += callInFinally(v);
        }
    }

    return s;
}

function main() {
    assert(viaContinue() == 6, "continue through a finally");
    assert(viaBreak() == 21, "break through a finally");
    assert(viaTwoJumps() == 15, "two continues through one finally");
    assert(viaLabelledContinue() == 1212, "a labelled continue through two finallys");
    assert(viaReturn(3) == 3, "return through a finally");
    assert(viaCallInFinally() == 18, "a call in the finally");
    assert(jumpInFinally() == 1, "a continue through a finally that breaks");
    print("done.");
}
