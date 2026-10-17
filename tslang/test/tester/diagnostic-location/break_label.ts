// A continue or break naming a label no statement around it has compiled with no error.

function main() {
    outer: for (let i = 0; i < 3; i++) {
        const f = () => {
            for (;;) {
                break outer;
            }
        };
        break outer;
    }
}
