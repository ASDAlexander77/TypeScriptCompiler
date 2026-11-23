// -mm=own --own-fallback=count, own only (POSIX twin of
// own_fallback_thread_word.ts): own_fallback_thread_pthread.ts's race, and after each
// join its child's header word read through the child's address. Both threads gave the child a second
// owner 100000 times and gave it back; the walk at Shared.share marked it shared, so each change was
// atomic and the word is the shared bit alone: no extra owner, no WeakRef. Counted plainly, changes
// were lost and the word held extra owners left over, or the child was freed while held and the word
// was the allocator's. (Under gc there is no header word, so this is not run against gc.)
@cdecl declare function pthread_create(thread: Opaque, attributes: Opaque | null, start: (arg: Opaque) => Opaque, arg: Opaque): s32;
declare function pthread_join(thread: u64, result: Opaque | null): s32;

class Child {
    constructor(public v: number) {}
}

class Node {
    constructor(public v: number, public child: Child) {}
}

class Holder {
    node: Node | null = null;
    child: Child | null = null;
}

// what the thread runs, and what it got: as the default library's ThreadStart
class ThreadStart {
    sum: number = 0;
    constructor(public fn: () => number) {}

    run() {
        this.sum = this.fn();
    }
}

function entry(arg: Opaque): Opaque {
    const start = <ThreadStart>arg;
    start.run();
    return arg;
}

function race(r: number): number {
    const keep = new Holder();
    const n = new Node(r, new Child(r + 1));
    const childAt = <u64><Opaque>n.child;
    keep.node = n;

    // nothing is allocated on the thread: under gc it is not one the collector knows
    const onThread = new Holder();
    const start = new ThreadStart(() => {
        let sum: number = 0;
        for (let i = 0; i < 100000; i++) {
            onThread.child = n.child;
            sum += n.v + onThread.child!.v;
        }

        onThread.child = null;
        return sum;
    });

    Shared.share(start);
    let thread: u64 = 0;
    assert(pthread_create(<Opaque>Ref(thread), null, entry, <Opaque>start) == 0, "pthread_create");

    const onMain = new Holder();
    let sum: number = 0;
    for (let i = 0; i < 100000; i++) {
        onMain.child = keep.node!.child;
        sum += keep.node!.v + onMain.child!.v;
    }

    onMain.child = null;
    keep.node = null;

    assert(pthread_join(thread, null) == 0, "pthread_join");

    // the node still holds the child (the closure's box holds the node until `start` goes)
    const word = Deref(<Ref<s64>><Opaque>(childAt - 8));
    assert(word != 0 && (word << 1) == 0, "the child's header word is its shared bit alone");
    return sum + start.sum;
}

function main() {
    let total: number = 0;
    for (let r = 0; r < 20; r++) {
        total += race(r);
    }

    print(total);
    print("done.");
}
