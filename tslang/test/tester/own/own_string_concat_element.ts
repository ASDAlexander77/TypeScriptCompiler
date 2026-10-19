// -mm=own: a concatenation reads its operands into a string of its own, so a borrowed element
// may be overwritten with what was made from it (`v[i] = v[i] + s`). And `assert` only reads its
// message, which may borrow an element.
export class Headers {
    values: string[] = [];

    append(i: int, value: string) {
        if (i < this.values.length) {
            this.values[i] = this.values[i] + ", " + value;
        } else {
            this.values.push(value);
        }
    }
}

function check(c: boolean, data: string[]) {
    assert(c, data[0]);
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i);
    }
}

function main() {
    let t = 0;
    for (let i = 0; i < 10000; i++) {
        const h = new Headers();
        h.append(0, "a");
        h.append(0, "b");
        churn();
        h.append(0, "c");
        check(h.values[0] == "a, b, c", h.values);
        t += h.values[0].length;
    }

    assert(t == 70000);
    print("done.");
}
