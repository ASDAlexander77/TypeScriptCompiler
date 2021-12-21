interface ReturnVal {
    something(): void;
}

function run(options: { something?(): void }, val: ReturnVal) {
    const something = options.something ?? val.something;
    something();
}

function main() {
    run({ something() { print("something"); } }, null);
    print("done.");
}
