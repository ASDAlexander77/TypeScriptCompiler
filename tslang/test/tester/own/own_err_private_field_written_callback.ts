// TSLANG_ISSUES 73: a private field another method writes may be overwritten by a callback the call
// runs, so a borrow of it across the call is refused.
const retained: ((n: int) => void)[] = [];

class Sample {
    frames = 0;
    callback: ((n: int) => void) | null = null;
    run(count: int): int {
        for (let i = 0; i < count; i++) {
            const callback = this.callback;
            if (callback) {
                callback(i);
            }
            this.frames++;
        }
        return this.frames;
    }
}

class App {
    total = 0;
    constructor(private sample: Sample) {}
    reset() {
        this.sample = new Sample();
    }
    frame(n: int) {
        this.total += n;
    }
    run(): int {
        const frame: (n: int) => void = this.frame;
        retained.push(frame);
        this.sample.callback = frame;
        const frames = this.sample.run(3);
        this.sample.callback = null;
        return frames + this.total;
    }
}

function main() {
    const app = new App(new Sample());
    assert(app.run() == 3 + 0 + 1 + 2);
    print("done.");
}
