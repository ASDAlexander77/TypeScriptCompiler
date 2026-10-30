// -mm=own copies an `any` rather than give it a second owner, as it copies a string (spec 71): a
// string boxed twice (`f(s); f(s)`) boxes a copy each time, a fresh box given to two calls is two
// boxes, and a @sharedElements container may hold `any`. A box's copy is a new box; a string in it is
// copied too, and a number is copied with the box. Each value is read after churn() has reused
// anything freed too early.
@sharedElements
class Bag<T> {
    items: T[] = [];

    add(item: T) {
        this.items.push(item);
    }

    has(item: T) {
        for (const i of this.items) {
            if (i == item) return true;
        }

        return false;
    }
}

class Keeper {
    kept: any[] = [];

    keep(a: any) {
        this.kept.push(a);
        return true;
    }
}

function churn() {
    const hold: string[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push("churn" + i);
    }
}

function main() {
    // the same string boxed into two calls that keep what they are given
    const k = new Keeper();
    const s = "Hel" + "lo";
    assert(k.keep(s));
    assert(k.keep(s));
    churn();
    assert(<string>k.kept[0] == "Hello" && <string>k.kept[1] == "Hello", "two boxes of one string");
    assert(s == "Hello", "the string itself");

    // a fresh box each time
    assert(k.keep("Wor" + "ld"));
    assert(k.keep("Wor" + "ld"));
    churn();
    assert(<string>k.kept[2] == "World" && <string>k.kept[3] == "World", "fresh boxes");

    // a container of `any`: strings and numbers
    const bag = new Bag<any>();
    const word = "w" + 1;
    bag.add(word);
    bag.add(word);
    bag.add(42);
    churn();
    assert(bag.items.length == 3, "three items");
    assert(bag.has("w1"), "a string item");
    assert(bag.has(42), "a number item");
    assert(word == "w1", "the string added twice");

    print("done.");
}
