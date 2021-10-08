let s2: string;
let xyz = 12;

function pause(ms: number): void {}

namespace control {
    export function runInBackground(a: () => void): void {
        a();
    }
}

function main() {
    print("test top level code");

    let r = 0;

    //s2 = "";
    for (let i = 0; i < 3; i++) {
        // TODO: this code breaks execution, i think because of variable i or copy in stack which is cleared after exiting
        let copy = i; // bug is here, when you put 'auto isCaptured = varOp.captured().hasValue() && varOp.captured().getValue();' with 'auto isCaptured = true;'
	// TODO: so using Malloc is allocating wrong byte size?
        control.runInBackground(() => {
		pause(r);
        });
    }

    print("done.");
}
