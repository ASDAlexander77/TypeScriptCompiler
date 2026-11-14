// TSLANG_ISSUES 50: a value moved into a field of a parameter is a borrow of that field after the
// move; the parameter given away into a field that is then cleared frees it, and the value with
// it, so the use after is refused (v0.0-pre-alpha127 took it and printed 0).
class C { x: int = 5; }
class H { c: C | null = null; }
class K {
    h: H | null = null;
    run(h: H) {
        const c = new C();
        h.c = c;
        this.h = h;
        this.h = null;
        print(c.x);
    }
}
function main() { new K().run(new H()); }
