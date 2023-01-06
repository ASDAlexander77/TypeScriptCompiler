function testLoopScope() {
    for (let i = 0; i < 3; ++i) {
        let val: number;
        // TODO:
        assert(val === undefined, "loopscope");
        val = i;
    }
}

function main() {
    testLoopScope();
    print("done.");
}