// An unlabeled `break` with no loop or switch around it in its function: that error, not "failed to legalize
// operation 'ts.Break'" (TSLANG 192).
function main() {
    while (true) {
        const f = () => {
            break;
        };
        f();
    }
}
