// -mm=own: a closure that escapes owns the object it captured, and gives it back when the closure
// goes (`c.onTick = null`). A use of the object after the capture read it freed: an error.

class Counter {
    onTick: ((n: number) => void) | null = null;
    count = 0;

    tick() {
        this.count++;
        if (this.onTick !== null) {
            this.onTick(this.count);
        }
    }
}

class Sink {
    seen = 0;
}

function main() {
    const c = new Counter();
    const s = new Sink();
    c.onTick = (n: number) => {
        s.seen += n;
    };
    c.tick();
    c.onTick = null;
    print(s.seen);
    print("done.");
}
