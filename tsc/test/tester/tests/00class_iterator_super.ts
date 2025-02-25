class B9 {
    g() {}
}

class C9 extends B9 {
    async * f() {
        super.g();
    }
}

print("done.");