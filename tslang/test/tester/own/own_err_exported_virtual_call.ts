// -mm=own, phase 4 rejects: `S` is exported, so another module may extend it and override
// `area` with one that overwrites `color`. The virtual call has candidates this module cannot see.
export class S {
    color: string = "red";
    w: number = 2;
    area() {
        return this.w * 2;
    }

    describe() {
        return `${this.color} area=${this.area()}`;
    }
}

function main() {
    print(new S().describe());
}
