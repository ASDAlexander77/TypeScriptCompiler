// -mm=own rejects: `App`'s constructor gives `this` to a call, and `rebind` writes the field it keeps
// `sample` in, so that field is not one the call cannot overwrite (TSLANG 76): `sample`, used after
// `new`, was moved.
@final
class Sample {
    options: string[] = [];

    add(option: string) {
        this.options.push(option);
    }
}

@final
class App {
    constructor(private sample: Sample) {
        this.reset();
    }

    reset() {}

    rebind(sample: Sample) {
        this.sample = sample;
    }
}

function main() {
    const sample = new Sample();
    const app = new App(sample);
    sample.add("-slow");
    app.rebind(new Sample());
}
