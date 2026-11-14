// -mm=own rejects: `rcp` calls `f`, a function value it captured, and `swap` gives `f` a function that
// empties `this.values`: the call is not known (TSLANG 82 resolves a captured function value only when
// every store into it is a function's symbol, and no closure writes it), so `a` may be freed.
class P {
    values: number[] = [2.0];

    calc(x: number): number {
        let f = (y: number): number => y + 1;
        const swap = () => {
            f = (y: number): number => {
                this.values = [];
                return y;
            };
        };
        const rcp = (y: number): number => f(y * 2);
        swap();
        return rcp(x);
    }

    run() {
        const a = this.values;
        this.calc(1.0);
        print(a[0]);
    }
}

function main() {
    new P().run();
}
