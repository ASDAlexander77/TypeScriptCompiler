// -mm=own refuses a `var` holding what a function returns from its rest parameter (00as in the
// corpus); what this test pins is that the message names `h` without --di, where it said 'this value'.
function tag(...x: string[]): string {
    return x[0];
}

function main() {
    var h = tag("a" + 1, "b");
    print(h);
}
