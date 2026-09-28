// --export=all exports the module's variables, not a function's: those were declared to
// importers as `@dllimport` globals in a `.f_<function>` namespace, which no library defines.
export class Node {
    items: number[] = [];

    get total(): number {
        let t = 0;
        for (const c of this.items) {
            t += c;
        }

        return t;
    }

    sum(): number {
        let u = 0;
        for (let i = 0; i < this.items.length; i++) u += this.items[i];
        return u;
    }
}

export function f() {
    let w = 1;
    return w;
}

let moduleLevel = 5;
