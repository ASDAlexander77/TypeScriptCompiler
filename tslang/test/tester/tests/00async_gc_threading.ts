// A coroutine is resumed on one of the async runtime's pool threads, and under `-mm=gc` its frame
// and everything its body builds come from the collector. Boehm allocates without taking a lock
// until something tells it there is more than one thread: `GC_need_to_lock` starts FALSE and its
// LOCK()/UNLOCK() expand to nothing. So a worker resuming a coroutine and the awaiting thread
// walked the same free lists at the same time with no lock at all, and the heap eventually said
// so - a fault, not a wrong answer. `GC_allow_register_threads` in the runtime's GC init sets that
// flag; the workers also register themselves, so a collection can suspend them and scan their
// stacks rather than free what only their registers still point at.
//
// `rc` and `none` were never affected: their coroutine frames go to the CRT heap, which locks
// whatever the program believes about itself. See docs/reference-counting-evaluation.md 9.57.
//
// This is a race, so it is a rate, not a certainty - both threads have to be inside the allocator
// together. Against the unfixed runtime this shape faults about 6 runs in 20 at `-O3` and 6 in 10
// at `-O0`; smaller ones are much weaker (60k iterations: 2 in 20), which is why the count is what
// it is. Each iteration allocates on both sides, so the two are in the allocator at the same time
// rather than taking turns.

class Node {
    v: number;

    constructor(v: number) {
        this.v = v;
    }
}

async function allocatesOnTheWorker(base: number): number {
    let node = new Node(base);

    return node.v;
}

function main() {
    let total = 0.0;
    for (let i = 0; i < 250000; i++) {
        total = total + await allocatesOnTheWorker(1.0);

        // The awaiting side allocates too.
        let mine = new Node(i);
        total = total + mine.v * 0.0;
    }

    assert(total == 250000.0, "every awaited body ran and gave its value back");

    print("done.");
}
