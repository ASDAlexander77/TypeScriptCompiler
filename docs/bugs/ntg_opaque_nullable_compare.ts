// Compiling a comparison of an Opaque with an `Opaque | null` field crashes the compiler with a
// stack overflow (exit code 0xC00000FD), printing nothing. Expected output: done.
class State {
    current: Opaque | null;

    constructor() {
        this.current = null;
    }

    set(next: Opaque): boolean {
        if (next != this.current) {
            this.current = next;
            return true;
        }
        return false;
    }
}

const s = new State();
print("done");
