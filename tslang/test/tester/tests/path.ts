export type Maybe<T> = null | undefined | T;

export interface Path {
    readonly prev: Path | undefined;
    readonly key: string | number;
    readonly typename: string | undefined;
}

/**
 * Given a Path and a key, return a new Path containing the new key.
 */
export function addPath(
    prev: Readonly<Path> | undefined,
    key: string | number,
    typename: string | undefined,
): Path {
    return { prev, key, typename };
}

/**
 * Given a Path, return an Array of the path keys.
 */
export function pathToArray(
    path: Maybe<Readonly<Path>>,
): (string | number)[] {
    let curr = path;
    let flattened = <typeof curr.key[]>[];
    while (curr) {
        flattened.push(curr.key);
        curr = curr.prev;
    }
    //flattened.reverse();
    return flattened;
}

function main() {
    let pathArray = pathToArray({
        key: "path",
        prev: undefined,
        typename: undefined,
    });
    for (let x of pathArray) {
        print(x);
    }

    print("done.");
}