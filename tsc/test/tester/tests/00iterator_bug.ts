class MatchResults
{
    constructor() {
    }    

    [index: number]: string;

    get(index: number) {
        return "";
    }

    get length() {
        return 0;
    }
}

function exec(): MatchResults | null {
    return new MatchResults();
}

type IterateResult<T> = { value: T, done: boolean };

interface Iterator<T> {
    next: () => IterateResult<T>;
}

function *matchAll(): Iterator<string[]> {

    while (true)
    {
        const result = exec();
        if (result == null)
            break;

        yield [...result];
    }
}

for (const v of matchAll()) { print("ok."); break; }

print("done.");