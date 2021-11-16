interface Ray {
    start: number;
    dir: number;
}

function test(v: Ray) {
    // BUG:
    print("v: ", v.start, v.dir);
}


function main() {
    print("start...");
    let p = 1.0;
    test({ start: 1.0, dir: p });
    print("done.");
}
