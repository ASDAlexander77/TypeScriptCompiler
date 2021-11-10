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

// TODO: not implemented
/*
interface X {
    a: (p: this) => this;
}

function main()
{
    let x: X = {
        a(p: X) {
            return p;
        }
    }

    let y = x.a(x);
}
*/

function main() {
    assert(s("sss"));

    let x = new X();
    assert(x.a(x).v == 10);

    print("done.")
}