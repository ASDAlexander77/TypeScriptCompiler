// TSLANG 213 (140): an `await` in a `finally` block. The finally is code of its own after the try, as a finally that
// yields is, run once on every way out of the try: its end, a `return` (its value kept for after the finally), a
// `break` or `continue`, a throw (thrown again after it).

let trail = "";

async function v(n: number): Promise<number> {
    return n;
}

async function afterTry(): Promise<number> {
    let r: number = 0;
    try {
        r = 1;
    } finally {
        r += await v(1);
    }

    return r;
}

async function returnThrough(x: number): Promise<number> {
    try {
        if (x > 0) return x * 10;
        trail += "b-body;";
    } finally {
        trail += "b-fin" + (await v(x)) + ";";
    }

    return -1;
}

async function fromCatch(): Promise<string> {
    try {
        throw "boom";
    } catch (e) {
        trail += "c-catch;";
        return "caught";
    } finally {
        await v(0);
        trail += "c-fin;";
    }
}

async function nested(): Promise<number> {
    try {
        try {
            return 7;
        } finally {
            trail += "d-inner" + (await v(1)) + ";";
        }
    } finally {
        trail += "d-outer" + (await v(2)) + ";";
    }
}

async function inLoop(): Promise<number> {
    let s = 0;
    for (let i = 0; i < 4; i++) {
        try {
            if (i == 1) continue;
            if (i == 3) break;
            s += i;
        } finally {
            s += await v(100);
        }
    }

    return s;
}

async function rethrown(): Promise<string> {
    try {
        try {
            throw "inner";
        } finally {
            trail += "f-fin" + (await v(3)) + ";";
        }
    } catch (x) {
        return "f-caught:" + x;
    }
}

async function awaitedReturn(): Promise<string> {
    let name = "s";
    try {
        return name + (await v(5));
    } finally {
        name = "changed";
        await v(6);
    }
}

async function noAnnotation(x: number) {
    try {
        if (x > 0) return x * 2;
    } finally {
        trail += "n" + (await v(1)) + ";";
    }

    return -1;
}

function sideEffect(): void {
    trail += "se;";
}

async function returnsNothing(): Promise<void> {
    try {
        return sideEffect();
    } finally {
        await v(3);
        trail += "vdc;";
    }
}

const arrow = async (x: number): Promise<string> => {
    try {
        return "a" + x;
    } finally {
        trail += "arrow" + (await v(4)) + ";";
    }
};

class K {
    base = 10;
    async m(x: number): Promise<number> {
        try {
            return this.base + x;
        } finally {
            trail += "m" + (await v(5)) + ";";
        }
    }
}

function main() {
    assert((await afterTry()) == 2);
    assert((await returnThrough(2)) == 20);
    assert((await returnThrough(0)) == -1);
    assert((await fromCatch()) == "caught");
    assert((await nested()) == 7);
    assert((await inLoop()) == 402);
    assert((await rethrown()) == "f-caught:inner");
    assert((await awaitedReturn()) == "s5");
    assert((await noAnnotation(3)) == 6);
    await returnsNothing();
    assert((await arrow(7)) == "a7");
    assert((await new K().m(1)) == 11);

    let r = 0;
    try {
        r = 1;
    } finally {
        r += await v(41);
    }

    assert(r == 42);
    assert(trail == "b-fin2;b-body;b-fin0;c-catch;c-fin;d-inner1;d-outer2;f-fin3;n1;se;vdc;arrow4;m5;");
    print("done.");
}
