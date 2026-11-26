// TSLANG 201:
// - a function called where it is made (`(function () { return f; })()`) given a type: the type was the function's
//   own, not its value's ("invalid cast from (p0: string) => string to string")
// - a function value given as one whose parameter is `any` (`(v: string) => string` as `(v: any) => string`), and one
//   returning an integer as one returning a number (`() => 0` as `(v: any) => number`): an adapter converts them
//   (they were "argument #1 does not match", "it takes 0 argument(s)")
var fromCall: (v: any) => string = (function () {
    return function (v: string) {
        return v + "!";
    };
})();

var zero: (v: any) => number = (function () {
    return function () {
        return 0;
    };
})();

function main() {
    assert(fromCall("ok") == "ok!", "a function called where it is made");
    const twice: (v: any) => number = (n: number) => n * 2;
    assert(twice(21) == 42, "an any parameter adapted");
    assert(zero(1) + 0.5 == 0.5, "an integer result read as a number");
    print("done.");
}
