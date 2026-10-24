// An imported @value class is nominal, as a local one is: an object literal of its shape is not one.
import './export_value_types'

function takeV(v: V) {
    return v.x;
}

function main() {
    print(takeV({ x: 1 }));
}
