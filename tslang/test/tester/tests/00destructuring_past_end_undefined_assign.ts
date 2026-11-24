// The assignment form: an element past the end of an array value, without a default, gives a target that can hold
// undefined (`let t: N | undefined; [s, t] = xs`) undefined, as node has it (it read whatever followed the array's
// data). gc, rc and none (own refuses storing a borrowed element into an outer `let`, as before).
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function sh(o: N | undefined | null): string {
    return o === undefined ? "u" : o === null ? "null" : "N" + o.x;
}

function main() {
    let out = "";
    for (let i = 0; i < 50; i++) {
        let s: N | undefined = mk(9);
        let t: N | undefined = mk(9);
        [s, t] = [mk(i)] as N[];
        let w: N | null | undefined = mk(9);
        const none: N[] = [];
        [w] = none;
        out = sh(s) + " " + sh(t) + " " + sh(w);
    }

    print(out);
    assert(out == "N49 u u", "assign");
    print("done.");
}
