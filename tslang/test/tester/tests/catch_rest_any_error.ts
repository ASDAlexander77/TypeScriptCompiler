// TSLANG 204: an error by design (it failed with "failed statement" alone)
function main() {
    try {
        throw { x: 1 };
    } catch ({ ...rest }) {
        print("caught");
    }
    print("done.");
}
