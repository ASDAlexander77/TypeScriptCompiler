// `a: b: loop`: both labels name the loop. `continue a` goes on to the loop's next iteration, as
// `continue b` does - it used to restart the loop from its initializer, skipping the incrementor.

function for_continue_outer_label() {
    let s = "";
    a: b: for (let i = 0; i < 3; i++) {
        if (i == 1) continue a;
        s += i;
    }

    return s;
}

function for_continue_inner_label() {
    let s = "";
    a: b: for (let i = 0; i < 3; i++) {
        if (i == 1) continue b;
        s += i;
    }

    return s;
}

function for_continue_from_nested_loop() {
    let s = "";
    a: b: c: for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            if (j == 1) continue a;
            if (i == 2) break b;
            s += i;
            s += j;
        }
    }

    return s;
}

function while_continue_outer_label() {
    let s = "";
    let i = 0;
    a: b: while (i < 4) {
        i++;
        if (i == 2) continue a;
        if (i == 4) break a;
        s += i;
    }

    return s;
}

function do_continue_outer_label() {
    let s = "";
    let i = 0;
    a: b: do {
        i++;
        if (i == 2) continue a;
        s += i;
    } while (i < 3);

    return s;
}

function for_of_continue_outer_label() {
    let s = "";
    a: b: for (const v of [1, 2, 3]) {
        if (v == 2) continue a;
        s += v;
    }

    return s;
}

// generating `[1, 2, 3]`'s iterator generated loops of its own, which took the label before the
// `for` the loop is made into existed: `continue b` then found no loop and crashed the compiler
function for_of_one_label() {
    let s = "";
    b: for (const v of [1, 2, 3]) {
        if (v == 2) continue b;
        s += v;
    }

    const arr: number[] = [4, 5, 6];
    c: for (const v of arr) {
        if (v == 5) continue c;
        if (v == 6) break c;
        s += v;
    }

    return s;
}

// a function inside the loop has labels of its own: its `a` is not another name for the outer `b`
function nested_function_label() {
    let s = "";
    a: b: for (let i = 0; i < 2; i++) {
        const f = () => {
            let r = "";
            a: for (let j = 0; j < 3; j++) {
                if (j == 1) continue a;
                r += j;
            }

            return r;
        };

        s += f();
    }

    return s;
}

function block_chain() {
    // a chain that ends in a block, not a loop, keeps its labels
    let s = "";
    a: b: {
        s += "x";
        if (s.length < 3) continue a;
    }

    return s;
}

function main() {
    assert(for_continue_outer_label() == "02", "for, continue a");
    assert(for_continue_inner_label() == "02", "for, continue b");
    assert(for_continue_from_nested_loop() == "0010", "for, continue a from a nested loop");
    assert(while_continue_outer_label() == "13", "while, continue a");
    assert(do_continue_outer_label() == "13", "do, continue a");
    assert(for_of_continue_outer_label() == "13", "for of, continue a");
    assert(for_of_one_label() == "134", "for of, one label");
    assert(nested_function_label() == "0202", "function in the loop");
    assert(block_chain() == "xxx", "block");

    print("done.");
}
