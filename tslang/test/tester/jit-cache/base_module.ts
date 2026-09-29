function makeStart() {
    return 10 + 5;
}

// not a constant: set by the module's own initializer, which has to run before its importers'
export let counter = makeStart();

export function bump() {
    counter++;
    return counter;
}

export class Base {
    static created = 0;

    constructor() {
        Base.created++;
    }

    hello() {
        return "base";
    }
}
