// TSLANG 204: an error by design (it was an internal error)
class f {}

function main() {
    const s = "" + f;
    print(s);
}
