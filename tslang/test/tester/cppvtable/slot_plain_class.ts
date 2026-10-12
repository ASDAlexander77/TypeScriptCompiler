// @slot is for a method of a @cpp class: an ordinary class has a vtable of tslang's own.
class Plain {
    @slot(0) first(): s32 {
        return 1;
    }
}

function main() {
}
