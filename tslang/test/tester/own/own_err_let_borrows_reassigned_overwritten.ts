// TSLANG_ISSUES 56: a `let` given only fields borrows each until the next assignment - but the field
// it holds now, written before the use, frees what it holds, so the use is refused.
class Pipe {
    id: int = 3;
}

class K {
    a: Pipe = new Pipe();
    b: Pipe = new Pipe();
    f = true;
    run(): int {
        let l = this.a;
        if (this.f) {
            l = this.b;
        }
        this.b = new Pipe();
        return l.id;
    }
}

function main() { print(new K().run()); }
