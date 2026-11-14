// TSLANG_ISSUES 57 (-mm=own), from native_ts_graphics' core/octomap_common.ts: an exported merge sort
// that keeps two `let`s over its parameters and swaps them each pass - `let src = order; let dst =
// scratch; ... const swap = src; src = dst; dst = swap;`. They only ever hold the caller's arrays, so
// they own nothing, and the function keeps no parameter.

export function sortDescending(order: int[], key: number[], scratch: int[]) {
    const n = order.length;
    let width = 1;
    let src = order;
    let dst = scratch;
    let inScratch = false;
    while (width < n) {
        for (let left = 0; left < n; left += 2 * width) {
            const mid = left + width < n ? left + width : n;
            const right = left + 2 * width < n ? left + 2 * width : n;
            let i = left;
            let j = mid;
            let k = left;
            while (i < mid && j < right) {
                if (key[src[j]] > key[src[i]]) {
                    dst[k] = src[j];
                    j++;
                } else {
                    dst[k] = src[i];
                    i++;
                }

                k++;
            }

            while (i < mid) {
                dst[k] = src[i];
                i++;
                k++;
            }

            while (j < right) {
                dst[k] = src[j];
                j++;
                k++;
            }
        }

        const swap = src;
        src = dst;
        dst = swap;
        inScratch = !inScratch;
        width *= 2;
    }

    if (inScratch) {
        for (let i = 0; i < n; i++) {
            order[i] = src[i];
        }
    }
}

export function longer(a: int[], b: int[]): int {
    let s = a;
    if (b.length > a.length) {
        s = b;
    }

    return s.length;
}

function main() {
    const order = [0, 1, 2, 3, 4];
    const scratch = [0, 0, 0, 0, 0];
    sortDescending(order, [5.0, 1.0, 9.0, 3.0, 7.0], scratch);
    print(order[0], order[1], order[2], order[3], order[4]);
    assert(order[0] == 2 && order[1] == 4 && order[2] == 0 && order[3] == 3 && order[4] == 1);
    assert(longer([1], [1, 2, 3]) == 3 && longer([1, 2], [1]) == 2);
    print("done.");
}
