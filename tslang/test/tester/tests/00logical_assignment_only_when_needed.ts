// `a ??= b`, `a ||= b` and `a &&= b` assign only where the left side does not decide: `b` is not
// evaluated, and nothing is stored (a setter does not run), where it does. The value of the
// expression is `b` where it was assigned, `a` where it was not. Under -mm=own an optional parameter
// given a default this way (`results ??= []`) is a plain assignment, not a merge of the parameter's
// own value and the new one. Values are made at run time, in a loop, so a leak or a double release shows.
class Box {
    private _v: string | undefined;
    sets = 0;

    get v(): string | undefined {
        return this._v;
    }

    set v(value: string | undefined) {
        this.sets++;
        this._v = value;
    }
}

let evaluated = 0;
function make(text: string) {
    evaluated++;
    return text;
}

// the right side grows the array the element is in: the value is read from where the element is now
const grown: (number | undefined)[] = [undefined];
function grow() {
    for (let i = 0; i < 1000; i++) {
        grown.push(i);
    }

    return 5;
}

function withDefault(results?: number[]) {
    results ??= [];
    results.push(1);
    return results.length;
}

function orDefault(results: number[] | undefined) {
    results ||= [];
    results.push(2);
    return results.length;
}

function orCallback(f?: (a: number) => number) {
    f ||= (a: number) => a;
    return f(42);
}

function main() {
    for (let i = 0; i < 2000; i++) {
        const b = new Box();
        b.v ??= make("first" + i);
        b.v ??= make("second" + i);
        assert(b.v == "first" + i, "??= assigned once");
        assert(b.sets == 1, "??= ran the setter once");

        b.v &&= make("third" + i);
        assert(b.v == "third" + i && b.sets == 2, "&&= assigned");
        b.v ||= make("fourth" + i);
        assert(b.v == "third" + i && b.sets == 2, "||= did not assign");

        let s: string | undefined = undefined;
        const r1 = (s ??= "x" + i);
        const r2 = (s ??= "y" + i);
        assert(r1 == "x" + i && r2 == "x" + i && s == "x" + i, "the value of ??=");

        let t = "";
        const r3 = (t ||= "z" + i);
        assert(r3 == "z" + i && t == "z" + i, "the value of ||=");

        assert(withDefault() == 1 && withDefault([5, 6]) == 3, "a parameter given a default");
        assert(orDefault(undefined) == 1 && orDefault([5]) == 2, "a parameter given a default by ||=");
        assert(orCallback() == 42 && orCallback((a: number) => a + 1) == 43, "a callback given a default");
    }

    const v = (grown[0] ??= grow());
    assert(v == 5 && grown[0] == 5 && grown.length == 1001, "an element whose array the right side grew");

    assert(evaluated == 2000 * 2, "the right side runs only where it is assigned");
    print("done.");
}
