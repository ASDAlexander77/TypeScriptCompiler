// A branch of `c ? f : g` that is a function of untyped parameters, with nothing that receives the result to type
// them, is instantiated with `any` as a `let` given one is: "variable 'f' is referencing generic type". Given to
// a function-typed parameter, the branches take its types. (Not under -mm=own, where a `let` given a `?:` of two
// function values takes a second reference.)
function apply(f: (n: number) => boolean, v: number) { return f(v); }

function main() {
    let isWhile = true;
    var f = isWhile ? function (n) { return n > 0; } : function (n) { return n === 0; };
    assert(f(1) && !f(0));

    const constant = false;
    var h = constant ? function (n) { return n > 0; } : function (n) { return n === 0; };
    assert(h(0) && !h(1));

    let k: number = 0;
    const g = isWhile ? (x) => { k += x; } : (x) => { k -= x; };
    g(5);
    assert(k == 5);

    assert(apply(isWhile ? (n) => n > 1 : (n) => n < 1, 2));
    print("done.");
}
