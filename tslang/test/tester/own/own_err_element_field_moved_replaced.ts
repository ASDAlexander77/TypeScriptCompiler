// TSLANG_ISSUES 59: a value moved into a field of an array's element borrows that field, so the
// element replaced before a use of the value frees it.
class View { n = 1; }
class Frame { view: View | null = null; }

export function f(frames: Frame[]): int {
    const frame = frames[0];
    const view = new View();
    frame.view = view;
    frames[0] = new Frame();
    return view.n;
}

function main() { print(f([new Frame()])); }
