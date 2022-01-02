function toHex(this: number) {
}

function numberToString(n: ThisParameterType<typeof toHex>) {
}

function main() {
    let fiveToHex: OmitThisParameter<typeof toHex>;

    print("done.");
}