// -mm=own: a method that overwrites a field under the borrowed value - the array a borrow reads,
// replaced by a method of the object holding it - drops the borrow (OWN_DROP_FIELDS_ATTR_NAME),
// though the method is called on that object and not on the borrow's root.
class Cam {
    position: number[] = [0.0, 0.0];

    reset(): void {
        this.position = [1.0, 1.0];
    }
}

class Pass {
    camera: Cam = new Cam();
}

function churn() {
    const keep: number[][] = [];
    for (let i = 0; i < 2000; i++) {
        keep.push([77777.0, 77777.0]);
    }

    return keep.length;
}

function main() {
    const p = new Pass();
    const position = p.camera.position;
    p.camera.reset();
    churn();
    print(position[0]);
}
