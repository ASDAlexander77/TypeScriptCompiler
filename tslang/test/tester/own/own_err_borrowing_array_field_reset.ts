// TSLANG_ISSUES 55: an array literal of fields borrows them while it is used - but a field written
// before a read out of the array frees what the array still holds, so that read is refused.
class Pipe {
    id: int = 3;
}

class K {
    a: Pipe = new Pipe();
    b: Pipe = new Pipe();
    run(): int {
        const xs = [this.a, this.b];
        this.a = new Pipe();
        return xs[0].id;
    }
}

function main() { print(new K().run()); }
