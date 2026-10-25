// An array an element is taken out of is no keeper: what it holds may be released, so the push is
// the move it always was, and a bound method of the caller's object cannot be given up.
type Callback = (n: number) => void;
let retained: Callback[] = [];

class Registry {
    register(handler: Callback): void {
        retained.push(handler);
    }
}

class Counter {
    constructor(registry: Registry) {
        registry.register(this.onTick);
    }

    onTick(n: number): void {}
}

function main() {
    const c = new Counter(new Registry());
    retained.pop();
}
