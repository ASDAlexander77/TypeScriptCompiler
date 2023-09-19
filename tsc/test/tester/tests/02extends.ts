class Base {
    check<TProp extends this>(prop: TProp): boolean {
        return true;
    }
}

class Test extends Base {
    m() {
        this.check(this);
    }
}

function main() {
    print("done.");
}
