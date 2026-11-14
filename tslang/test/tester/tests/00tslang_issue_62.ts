// TSLANG_ISSUES 62 (-mm=own), from native_ts_graphics' examples/ray_tracing_extended.ts: `const model =
// this.refractionModel; ... model.push(0.0);` - an array changed in place through a local that borrows
// the field holding it. The local's slot holds the field's array, so the field sees each change.

class V {
    n: int;
    constructor(n: int) {
        this.n = n;
    }
}

class Holder {
    model: number[] = [];
    items: V[] = [new V(1)];

    fill(): number {
        const model = this.model;
        if (model.length == 0) {
            for (let i = 0; i < 4; i++) {
                model.push(1.5);
            }
        }

        return model.length;
    }

    churn(): int {
        const items = this.items;
        items.push(new V(2));
        items.unshift(new V(0));
        items.splice(1, 1);
        const first = items.shift();
        const last = items.pop();
        return (first ? first.n : -1) * 10 + (last ? last.n : -1) + items.length * 100;
    }
}

function main() {
    const h = new Holder();
    assert(h.fill() == 4 && h.fill() == 4 && h.model.length == 4);
    assert(h.churn() == 2 && h.items.length == 0);
    print("done.");
}
