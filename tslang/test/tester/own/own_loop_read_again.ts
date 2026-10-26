// -mm=own: a value read in a loop and given to a call later in the same iteration is that
// iteration's. A `continue` before the call goes around the loop, where the slot is given its next
// value, and back to the read: that is a new read, not one the call is given (findBetween). The
// strings yielded are the loop's own; `keep` takes each.
function* words() {
    yield "a" + 1;
    yield "b" + 2;
    yield "c" + 3;
}

class Keep {
    items: string[] = [];

    add(s: string) {
        this.items.push(s);
    }
}

function skips(s: string) {
    return s == "b2";
}

function main() {
    const k = new Keep();
    for (const w of words()) {
        if (skips(w)) continue;
        k.add(w);
    }

    assert(k.items.length == 2 && k.items[0] == "a1" && k.items[1] == "c3");
    print("done.");
}
