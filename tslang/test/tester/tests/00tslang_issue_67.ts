// TSLANG_ISSUES 67-68 (-mm=own), from native_ts_graphics:
// 67: examples/nri/rt_shadows_nri.ts, `const view = buffer.createConstantView(...); if (view)
//     this.constantViews.push(view); return view;` - a value pushed only where it is not null, then used
//     where the paths meet: the element pushed, or null.
// 68: examples/conservative_rasterization.ts, `ImGui.text(`...: ${this.yesNo(p[3])}`)` with `yesNo`
//     returning `v != 0.0 ? "yes" : "no"` - a union of string literals read as a `string`. It allows
//     those literals only: `"maybe"`, or a `string`, given to it is an error (TS2322).

class View {
    constructor(public n: int) {}
}

function make(n: int): View | null {
    return n > 0 ? new View(n) : null;
}

function yesNo(v: number) {
    return v != 0.0 ? "yes" : "no";
}

function show(s: string): int {
    return s.length;
}

class App {
    views: View[] = [];

    view(n: int): View | null {
        const view = make(n);
        if (view) {
            this.views.push(view);
        }
        return view;
    }

    count(n: int): int {
        const view = make(n);
        if (view) {
            this.views.push(view);
        }
        if (!view) {
            return -1;
        }
        return view.n;
    }

    fill(n: int): int {
        let total = 0;
        for (let i = 0; i < n; i++) {
            const view = make(i % 2);
            if (view) {
                this.views.push(view);
            }
            total += view ? view.n : 100;
        }
        return total;
    }

    yesNo(v: number) {
        return v != 0.0 ? "yes" : "no";
    }

    texts(p: number[]): int {
        let s = yesNo(p[0]);
        if (p[1] == 0.0) {
            s = "no";
        }
        return show(`a: ${yesNo(p[0])}`) + show(`b: ${this.yesNo(p[1])}`) + show("c: " + s) + show(yesNo(p[1]));
    }
}

function main() {
    const app = new App();
    const v = app.view(3);
    assert(v != null && v.n == 3 && app.view(0) == null);
    assert(app.count(4) == 4 && app.count(0) == -1);
    assert(app.fill(4) == 202 && app.views.length == 4);
    assert(app.texts([1.0, 0.0]) == 6 + 5 + 5 + 2);
    print("done.");
}
