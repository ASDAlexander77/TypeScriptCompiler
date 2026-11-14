// TSLANG 76: a constructor that keeps a parameter in a private field and gives `this` to a call
// (`this.reset()`): no other function writes the field, so the call cannot overwrite it, and the
// parameter, used after `new`, is read back from the field.
@final
class Sample {
    options: string[] = [];

    add(option: string) {
        this.options.push(option);
    }
}

@final
class App {
    private constants: number[] = [0, 0];

    constructor(private sample: Sample) {
        this.reset();
    }

    reset() {
        this.constants[0] = 4.0;
    }

    count(): int {
        return this.sample.options.length;
    }
}

function main() {
    const sample = new Sample();
    const app = new App(sample);
    sample.add("-slow");
    sample.add("-noUI");
    assert(app.count() == 2, "count");
    print("done.");
}
