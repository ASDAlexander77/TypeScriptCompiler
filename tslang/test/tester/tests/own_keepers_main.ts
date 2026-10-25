// The program half of own-keepers-objects.cmake: it registers a bound method of an object and a
// closure over a local with another module's `register`, which keeps them for good, from functions
// that return before they run. Reading the declaration's `@immortalizes`, -mm=own makes the object,
// the box and the cell immortal here, so they outlive their owners' releases.
import { register, runAll } from "./own_keepers_lib";

let results: number[] = [0.0, 0.0];

class Counter {
    count: number[] = [0.0];

    constructor() {
        register(this.onTick);
    }

    onTick(n: number): void {
        this.count[0] += n;
        results[0] = this.count[0];
    }
}

function setupBound(): void {
    const c = new Counter();
}

function setupClosure(): void {
    const total = [100.0];
    register((n: number) => { total[0] += n; results[1] = total[0]; });
}

function churn(): number {
    let sum = 0.0;
    for (let i = 0; i < 20000; i++) {
        const a = [i * 1.0, 2.0];
        sum += a[0];
    }

    return sum;
}

function main() {
    setupBound();
    setupClosure();
    churn();
    runAll(1);
    churn();
    const kept = runAll(1);
    print(results[0], results[1], kept);   // 2 102 2
    print("done.");
}
