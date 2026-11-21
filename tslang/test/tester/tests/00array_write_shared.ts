// A write of an element of an array held by a handle (`s.value[i] = v`, as the library's TypedArray.set
// does): under -mm=own it does not grow the array there, and so a method that writes one element is
// not taken for one that releases what its callers borrow ("a value that borrows 'this.sync' is used
// here after it may be released").

class Cells {
    private store = new Shared<int[]>([0, 0, 0]);
    public set(index: int, value: int) {
        this.store.value[index] = value;
    }

    public get(index: int) {
        return this.store.value[index];
    }
}

class Queue {
    sync = new Cells();
    close() {
        this.sync.set(2, 1);
    }
}

class Worker1 {
    q = new Shared<Queue>(new Queue());
    end() {
        this.q.value.close();
        this.q.value.close();
    }
}

function main() {
    const w = new Worker1();
    w.end();
    assert(w.q.value.sync.get(2) == 1, "written");
    print("done.");
}
