// A caught value cast to its class, given to a function own has no facts for (declared here, as a
// default library's virtual `instanceOf` is to its callers), and read after the call: the call is
// given the value, which it borrows, and nothing else holds the exception, so it cannot free it.
// Before, the call was taken to free the catch variable's value. Compiled only: nothing defines
// `checkCode`.
class Failure {
    constructor(public code: int) {}
}
function fail(c: int) {
    if (c > 0) throw new Failure(c);
}
declare function checkCode(f: Failure, c: int): boolean;
function caught(c: int) {
    let matched = false;
    try {
        fail(c);
    } catch (e) {
        const failure = <Failure>e;
        matched = checkCode(failure, c);
        matched = matched && failure.code == c;
    }
    return matched;
}
function main() {
    print(caught(0));
    print("done.");
}
