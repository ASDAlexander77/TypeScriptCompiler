// An unlabeled `continue` with no loop around it in its function (a `switch` is not one): that error, not "failed
// to legalize operation 'ts.Continue'" (TSLANG 192).
function main() {
    switch (1) {
        default:
            continue;
    }
}
