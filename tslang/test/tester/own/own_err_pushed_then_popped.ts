// TSLANG_ISSUES 52: a value pushed into an array that is a field is a borrow of its last element
// after the push - but `pop` releases that element, and the value with it, so the use after is
// refused.
class Tex {
    id: int = 7;
}

class K {
    texs: Tex[] = [];
    run(): int {
        const t = new Tex();
        this.texs.push(t);
        this.texs.pop();
        return t.id;
    }
}

function main() { print(new K().run()); }
