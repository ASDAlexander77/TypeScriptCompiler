// A member of a class declared in a function reads one of its locals: nothing would pass it to the
// member, so it is an error naming the variable (docs/classes-as-values-design.md §3).
function counter(start: number) {
    class Counter {
        n = 0;

        inc() {
            this.n += start;
            return this.n;
        }
    }

    return new Counter();
}

function main() {
    print(counter(2).inc());
}
