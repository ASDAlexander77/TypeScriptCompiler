// `++` and `--` on a get/set property call the setter with the getter's value plus or minus one, as `+= 1` does:
// a prefix operator gives the new value, a postfix one the old. It was an error, "'++'/'--' on a get/set accessor
// property is not supported".
class Counter {
    private n: number = 1;
    sets = 0;
    get v() { return this.n; }
    set v(x: number) { this.sets++; this.n = x; }

    static total: number = 10;
    static get all() { return Counter.total; }
    static set all(x: number) { Counter.total = x; }

    bump() {
        return this.v++;
    }
}

class Big extends Counter {
    override get v() { return super.v; }
    override set v(x: number) { super.v = x * 10; }
}

interface Valued {
    v: number;
}

function main() {
    const c = new Counter();
    const post = c.v++;
    assert(post == 1 && c.v == 2, "postfix gives the old value");
    const pre = ++c.v;
    assert(pre == 3 && c.v == 3, "prefix gives the new value");
    c.v--;
    --c.v;
    assert(c.v == 1 && c.sets == 4, "decrement, once through the setter each");

    Counter.all++;
    assert(Counter.all == 11, "static");

    assert(c.bump() == 1 && c.v == 2, "through this");

    const b: Counter = new Big();
    b.v++;
    assert(b.v == 20, "virtual setter");

    const valued: Valued = c;
    valued.v++;
    assert(c.v == 3, "through an interface");

    const o = {
        h: 5.0,
        get x() { return this.h; },
        set x(value: number) { this.h = value; }
    };
    o.x++;
    assert(o.h == 6, "object literal");

    print("done.");
}
