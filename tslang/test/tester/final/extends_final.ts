// A @final class cannot be extended: calls on it are direct, so an override would never run.
@final
class B {
    m() {
        return 1;
    }
}

class X extends B {}

function main() {
    print(new X().m());
}
