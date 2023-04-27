let p: Promise<number>;

async function fn(): Promise<number> {
    const i = await p; // suspend execution until 'p' is settled. 'i' has type "number"
    return 1 + i;
}

class C {
    async m(): Promise<number> {
        const i = await p; // suspend execution until 'p' is settled. 'i' has type "number"
        return 1 + i;
    }
    async get p(): Promise<number> {
        const i = await p; // suspend execution until 'p' is settled. 'i' has type "number"
        return 1 + i;
    }
}

function main() {
    const a = async (): Promise<number> => 1 + await p; // suspends execution.
    const a2 = async () => 1 + await p; // suspends execution. return type is inferred as "Promise<number>" when compiling with --target ES6
    const fe = async function (): Promise<number> {
        const i = await p; // suspend execution until 'p' is settled. 'i' has type "number"
        return 1 + i;
    }
}