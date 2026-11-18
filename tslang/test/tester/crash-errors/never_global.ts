function fail(message: string): never {
    throw message;
}

const unreachable: never = fail("never returns");

function read(): number {
    return unreachable;
}
