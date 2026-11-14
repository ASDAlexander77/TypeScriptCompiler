// TSLANG_ISSUES 48: a borrow of a field held across `this.factor()`, a `@final` method of a class
// another module defines: no override can run, so the call drops nothing the method's facts don't.
import './own_final_method_lib'
class App extends Base {
    total(): number {
        const item = this.items[1];
        const f = this.factor();
        return item.n * f;
    }
}
function main() { const t = new App().total(); print(t); assert(t == 4); print("done."); }
