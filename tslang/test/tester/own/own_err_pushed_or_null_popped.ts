// TSLANG_ISSUES 67: a value pushed only where it is not null borrows the element where the paths meet,
// so the element popped before the use frees it.
class View {
    constructor(public n: int) {}
}
function make(n: int): View | null {
    return n > 0 ? new View(n) : null;
}
class App {
    views: View[] = [];
    count(n: int): int {
        const view = make(n);
        if (view) {
            this.views.push(view);
        }
        this.views.pop();
        return view ? view.n : -1;
    }
}
function main() {
    const app = new App();
    print(app.count(4));
    print("done.");
}
