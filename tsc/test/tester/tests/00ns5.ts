namespace nn {
    class c1 {}
}

function main() {
    const c = new nn.c1();
    delete c;
    print("done.");
}
