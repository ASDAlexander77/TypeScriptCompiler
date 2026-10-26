// -mm=own, strings as values (spec 22): a string read out of a place, given to a call that may free
// it - a callback, which nothing can say `@nodrops` of - or to a call that keeps it, gets a copy. A
// tuple made of borrowed strings starts from copies of them. Each original is overwritten after, so
// a borrow instead of a copy reads the new value or freed memory.
class Bag {
    items: string[] = [];

    forEach(f: (s: string) => void) {
        for (let i = 0; i < this.items.length; i++) f(this.items[i]);
    }

    keep(s: string) {
        this.items.push(s);
    }

    fill(from: string[]) {
        for (let i = 0; i < from.length; i++) this.keep(from[i]);
    }
}

class P {
    a = "";
    b = "";
}

const p = new P();

function both(x: string, y: string) {
    p.a = x;
    p.b = y;
}

function pair(a: string[]): [string, number] {
    const s = a[0];
    return [s, 1];
}

function main() {
    // the callback overwrites the very element it was given
    const b = new Bag();
    b.items.push("a" + 1);
    let r = "";
    b.forEach((s) => {
        b.items[0] = "zzzzzzzzzzzzzzzz";
        r += s;
    });
    assert(r == "a1");

    // a keeping call takes a copy of each element; the source is the caller's still
    const src = ["b" + 2, "c" + 3];
    const k = new Bag();
    k.fill(src);
    src[0] = "yyyyyyyyyyyyyyyy";
    assert(k.items[0] == "b2" && k.items[1] == "c3" && src[1] == "c3");

    // a field's string given twice to a call that keeps both: each is a copy
    const h = new P();
    h.a = "s" + 1;
    both(h.a, h.a);
    h.a = "xxxxxxxxxxxxxxxx";
    assert(p.a == "s1" && p.b == "s1");

    // a tuple of a borrowed element
    const a = ["d" + 4];
    const t = pair(a);
    a[0] = "wwwwwwwwwwwwwwww";
    assert(t[0] == "d4" && t[1] == 1);

    print("done.");
}
