function s(v: any): v is string {
    return typeof v === "string";
}

class X {
    v = 10;
    a(p: this) {
        print("Hello...");
        print("Hello", this.v);
        print("Hello", p.v);
        return p;
    }
}

interface I {
    a: (p: this) => this;
}

function main() {
    assert(s("sss"));

    let x = new X();
    assert(x.a(x).v == 10);

    let x2: I = {
        a(p: I) {
            return p;
        }
    }

    let y = x2.a(x2);

    print("done.")
}