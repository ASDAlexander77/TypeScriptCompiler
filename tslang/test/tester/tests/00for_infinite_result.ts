// regression test: ForOpLowering's NoConditionOp branch (infinite for(;;))
// never assigned the loop-carried result values, so replaceOp ran with an
// empty ValueRange whenever the for(;;) had loop-carried results.
function testInfiniteForResult(): number {
    let sum = 0;
    let i = 0;
    for (;;) {
        if (i >= 5) {
            break;
        }

        sum = sum + i;
        i = i + 1;
    }

    return sum;
}

function main() {
    assert(testInfiniteForResult() == 10, "for(;;) with loop-carried result");
    print("done.");
}
