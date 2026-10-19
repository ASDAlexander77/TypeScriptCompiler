// -mm=own: a borrow of a field can be lent to a method of a @final class that destroys nothing:
// the call is direct, so its body's facts are known. Through the vtable another module's override
// might destroy anything (the default library's String wrapper lending `this.value` to RegExp).
@final
export class Re {
    test(s: string): boolean {
        return s.length > 2;
    }
}

export class W {
    constructor(private value: string) {}

    has(re: Re): boolean {
        return re.test(this.value);
    }
}

class Pad {
    constructor(public s: string) {}
}

function churn() {
    const junk: Pad[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Pad("junk" + i));
    }
}

function main() {
    const re = new Re();
    let t = 0;
    for (let i = 0; i < 10000; i++) {
        const w = new W(i % 2 == 0 ? "abc" + i : "a");
        churn();
        if (w.has(re)) {
            t++;
        }
    }

    assert(t == 5000);
    print("done.");
}
