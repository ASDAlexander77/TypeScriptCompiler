// -mm=rc: an argument read out of an array (`f4(arr[0], al[0])`, `al = [arr]`) given to a callee that shortens that
// array through its other name (`a.length = 0` gives back the element it drops) is held by the caller: `a.length = n`
// is an element write (writtenKinds) as `pop`, `shift` and `splice` are. rc read freed memory (3751995 against gc's
// 4498500). Alone in its module, as 00rc_argument_place_aliased's copy of it is held there for another reason.
class N {
    x: number;
    s: string;
    constructor(x: number) {
        this.x = x;
        this.s = "n" + x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

@noinline function ms(i: number): string {
    return "s" + i;
}

@noinline function take(o: N | null): number {
    return o ? o.x + o.s.length : 0;
}

@noinline function takes(s: string | null): number {
    return s ? s.length : 0;
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 16; i++) {
        junk.push(new N(i));
    }
}
function f4(p: N, a: N[]): number { a.length = 0; let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function main() {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const al = [arr]; total += f4(arr[0], al[0]);
    }
    print(total);
    assert(total == 44850, "total");
    print("done.");
}
