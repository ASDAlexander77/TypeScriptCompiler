// cycle_module_a and cycle_module_b import each other. Generating them used to never end.
import './cycle_module_b'

export function cycle_a_fn() { return 1; }
