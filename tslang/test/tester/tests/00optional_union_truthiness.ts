// `null | undefined | { ... }` with no value is false in a condition. The optional's default, a
// zeroed union whose type is "undefined", is not null, and it was read as true: the loop never ended.
// (path.ts got this type from the default library's Readonly<Path>.)

interface Path {
    readonly prev: Path | undefined;
    readonly key: string | number;
}

type PathLike = null | undefined | { key: string | number, prev: Path | undefined };

function pathToArray(path: PathLike): (string | number)[] {
    let curr = path;
    let flattened = <(string | number)[]>[];
    while (curr) {
        flattened.push(curr.key);
        curr = curr.prev;
    }
    return flattened;
}

function count(p: PathLike) {
    let n = 0;
    if (p) n++;
    if (!p) n += 10;
    return p ? n + 100 : n;
}

function main() {
    const keys = pathToArray({ key: "path", prev: { key: 2, prev: undefined } });
    assert(keys.length == 2);
    assert(keys[0] == "path");
    assert(keys[1] == 2);

    assert(pathToArray(undefined).length == 0);
    assert(pathToArray(null).length == 0);

    assert(count(undefined) == 10);
    assert(count(null) == 10);
    assert(count({ key: 1, prev: undefined }) == 101);

    print("done.");
}
