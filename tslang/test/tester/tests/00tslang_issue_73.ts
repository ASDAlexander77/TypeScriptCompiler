// TSLANG_ISSUES 73 (-mm=own), from native_ts_graphics' NRI examples: `constructor(private sample: NriSample)`
// and `this.sample.run(...)`, which runs callbacks into the app. A private field nothing writes after the
// constructor is not overwritten by them, so the borrow of it holds across the call.
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
