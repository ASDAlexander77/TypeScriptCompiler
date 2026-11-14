// TSLANG_ISSUES 53: a value given to a constructor that keeps it in a field is that field after the
// call - but a method of the object that overwrites the field frees it, so the use after is refused.
class S {
    n: int = 4;
}

class App {
    constructor(public s: S) {}
    reset() { this.s = new S(); }
}

function main() {
    const s = new S();
    const app = new App(s);
    app.reset();
    print(s.n);
}
