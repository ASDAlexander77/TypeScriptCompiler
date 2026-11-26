// An array pattern of an iterable (`const [a, b, c, d] = new Set([...])`): each element is read as
// `done ? undefined : next().value`. Under -mm=own that read was a ts.Optional of the field of next()'s
// result, which own does not follow to its receiver: once next() was known to return its result owned,
// the pattern was refused ("a value takes a second reference"), and DefaultLib's bindings.ts with it
// (release of 2026-11-26). Runtime strings, not literals: a literal is immortal and hides a wrong count.
class Bag {
    items: string[] = [];

    add(s: string) {
        this.items.push(s);
    }

    // the shape of DefaultLib's Set<T>.iter: an object literal whose next() reads the owner
    [Symbol.iterator]() {
        return {
            i: 0,
            owner: this,
            next() {
                if (this.i < this.owner.items.length) {
                    const value = this.owner.items[this.i];
                    this.i++;
                    return { value, done: false };
                }

                return { value: "", done: true };
            }
        };
    }
}

function* strings(k: number) {
    yield "g" + k;
    yield "h" + k;
}

function bagOf(k: number) {
    const bag = new Bag();
    bag.add("a" + k);
    bag.add("b" + k);
    bag.add("c" + k);
    return bag;
}

const k = 7;
const [ma, mb, mc, md] = bagOf(k);
assert(ma == "a7" && mb == "b7" && mc == "c7", "module level");
assert(md === undefined, "past the end at module level");

const [ga, gb, gc] = strings(k);
assert(ga == "g7" && gb == "h7" && gc === undefined, "generator at module level");

function local(i: number): number {
    const [a, b, c, d] = bagOf(i);
    return (a ?? "").length + (b ?? "").length + (c ?? "").length + (d === undefined ? 100 : 0);
}

function fewer(i: number): string {
    const [a] = bagOf(i);
    return a ?? "";
}

function generated(i: number): number {
    const [a, b, c] = strings(i);
    return (a ?? "").length + (b ?? "").length + (c === undefined ? 100 : 0);
}

function main() {
    assert(local(1) == 6 + 100, "local");
    assert(fewer(3) == "a3", "fewer slots than elements");
    assert(generated(1) == 4 + 100, "generator");

    let n = 0;
    for (let i = 0; i < 20000; i++) {
        n += local(i) + fewer(i).length + generated(i);
    }

    assert(n > 0, "many times");

    print("done.");
}
