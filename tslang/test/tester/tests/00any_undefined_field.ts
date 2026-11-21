// An object literal's field of `<any>undefined` or `<any>null`: the box is a block of its own, which the
// store into the object moves in. Under -mm=own it was taken for constant data (null or undefined seen
// through its casts), so the move was not checked and the temporary's release gave the box back as well
// as the object's: a double free, which wasm's checked heap and Linux caught (TSLANG 153).
function mk() {
    return { t: <any>undefined, n: 1 };
}

function mkNull() {
    return { t: <any>null, n: 2 };
}

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        const o = mk();
        const p = mkNull();
        total += o.n + p.n;
        if (o.t !== undefined || p.t !== null) total = -1000000;
    }

    assert(total == 3000, "fields");
    print("done.");
}
