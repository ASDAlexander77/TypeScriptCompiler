// -mm=own: an element of an array literal nested in another that a conditional expression's branch yields
// (`c ? [[mk(i)]][0][0] : null`): the outer literal is held past the conditional, as for a flat one
// (own_literal_temp_conditional), and given back at the end of the block with the inner one and what it holds. It
// leaked every evaluation before. The same program under rc (where the branch retains the element before the outer
// literal is given back, see rc_conditional_nested_literal) and gc.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(i));
    }
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 40; i++) {
        const nested: N | null = i % 4 == 0 ? [[mk(i)]][0][0] : null;
        if (nested) {
            total += nested.x;
        }

        const deep: N | null = i % 2 == 0 ? [[[mk(i), mk(1)]]][0][0][1] : null;
        if (deep) {
            total += deep.x;
        }

        churn();
    }

    print(total);
    assert(total == 180 + 20, "total");
    print("done.");
}
