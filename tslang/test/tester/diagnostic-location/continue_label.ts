// A continue or break naming a label no statement around it has compiled with no error.

function main() {
    outer: for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            if (j == 1) continue outer;
            continue nolabel;
        }
    }
}
