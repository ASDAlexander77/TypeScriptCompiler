// -mm=own, phase 5b rejects: two closures that escape cannot both own `n`.
let handlers: (() => number)[] = [];

// emptied, so no keeper: what it holds may be released (a keeper's would be immortal)
function reset() {
    handlers.length = 0;
}

function register() {
    let n: number = 0;
    handlers.push(() => ++n);
    handlers.push(() => n);
}

function main() {
    register();
    print(handlers.length);
}
