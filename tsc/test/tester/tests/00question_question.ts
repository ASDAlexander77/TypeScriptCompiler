// @strict-null false
interface ReturnVal {
    something(): void;
}

function run(options: { something?(): void }, val: ReturnVal) {
    const something = options.something ?? val.something;
    something();
}

let glb1 = false;
function main() {
    run({ something() { print("something"); glb1 = true; } }, null);
    assert(glb1);
    print("done.");
}
