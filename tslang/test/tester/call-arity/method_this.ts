class O {
    x = 1;

    m(this: O, a: number, b: number) {
        return a + b + this.x;
    }
}

function main() {
    const o = new O();
    print(o.m(1));
}
