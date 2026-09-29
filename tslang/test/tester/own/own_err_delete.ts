// -mm=own, phase 0 rejects: `delete` destroys the value while the slot still releases it at scope exit.
class C { v = 1; }
function main() {
    let c = new C();
    delete c;
    print("done.");
}
