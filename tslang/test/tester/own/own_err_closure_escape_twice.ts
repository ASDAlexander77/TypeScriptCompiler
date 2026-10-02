// -mm=own, phase 5b rejects: two closures that escape cannot both own `n`.
let handlers: (() => number)[] = [];

function register() {
    let n: number = 0;
    handlers.push(() => ++n);
    handlers.push(() => n);
}

function main() {
    register();
    print(handlers.length);
}
