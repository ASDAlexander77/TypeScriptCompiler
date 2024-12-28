function log(...data: string[]): void {
    printData(1, data);
}

function printData(fileNo: int, data: string[]): void {
    switch (data.length) {
        case 0:
            break;
        default:
            print1(fileNo, data[0]);
            for (let i = 1; i < data.length; i++) {
                print1(fileNo, " ");
                print1(fileNo, data[i]);
            }
    }

    print1(fileNo, "\n");
}

function print1(fileNo: int, data: string): void {
    // ...
    print(data);
}

const vf32: f32 = 12.00;
const boxed3: any = vf32;

log(vf32, boxed3);

print("done.");