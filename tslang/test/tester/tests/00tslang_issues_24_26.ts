// TSLANG_ISSUES 24 and 25, and the references 26 still allows.

// 24: a shift, `&`, `|` or `^` is ToInt32's signed 32-bit integer, as `int` is: an array literal holding
// `1 << 18` and an `int` is an int array, not one of 64-bit integers
const CAP = 1 << 18;
const MASK = 0xff & 0x0f;

class Grid {
    w: int = 290;

    header() {
        return [CAP, this.w, MASK | 1];
    }
}

// 25: an empty array literal against a non-empty one is an array of the other's elements
class S {
    scene: int = 4;

    check(): int {
        const expected = this.scene == 4 ? [1] : [];
        let sum = 0;
        for (const k of expected) {
            sum += k;
        }

        return sum;
    }

    nested(): int {
        const expected = this.scene == 2 ? [0, 2] : this.scene == 4 ? [1] : this.scene == 5 ? [2, 3] : [];
        let sum = 0;
        for (const k of expected) {
            sum += k;
        }

        return sum + expected.length * 10;
    }

    emptyFirst(): int {
        const expected = this.scene != 4 ? [] : [5, 6];
        return expected.length;
    }
}

// 26: a reference to a number laid out as the parameter's is accepted
function fillF32(dst: Ref<f32>) {
    return 1;
}

function fillInt(dst: Ref<int>) {
    return 2;
}

function main() {
    const header = new Grid().header();
    assert(header.length == 3 && header[0] == 262144 && header[1] == 290 && header[2] == 15, "values");
    assert(sizeof(header[0]) == 4, "an int array");

    const s = new S();
    assert(s.check() == 1, "the non-empty branch");
    assert(s.nested() == 11, "nested conditionals");
    assert(s.emptyFirst() == 2, "the empty literal first");
    s.scene = 3;
    assert(s.check() == 0 && s.nested() == 0 && s.emptyFirst() == 0, "the empty branch");

    const floats: f32[] = [1, 2, 3];
    const ints: int[] = [1, 2, 3];
    assert(fillF32(Ref(floats[0])) == 1 && fillInt(Ref(ints[1])) == 2, "references laid out alike");

    // asserted, a reference reads what it refers to as another type: the default library hashes a
    // number's bits so. 1.5 is 0x3FF8000000000000
    let d: number = 1.5;
    const halves = <Reference<i32>>Ref(d);
    assert(Deref(halves[0]) == 0 && Deref(halves[1]) == 0x3ff80000, "<Reference<i32>> of a number");
    const low = Ref(d) as Reference<i32>;
    assert(Deref(low) == 0, "as Reference<i32>");

    print("done.");
}
