#!/usr/bin/env bash
# Masks the run-to-run nondeterminism in tslang's emitted IR, so two builds can be diffed.
# The noise is hash-like IDs that change between runs of the same binary:
#   - 6+ digits after a letter/underscore inside a name: @s_<n>, %FH<n>, td_<n>_object
#   - FH followed by any number of digits: the function-hash suffix of generated names
#     (.objL10C13FH<n>, ..afL27C20FH<n>) is sometimes only 3-4 digits long (FH668, FH6184)
#   - 6+ digits between '.' and '..vtbl': @"Iterable<si32>.<n>..vtbl"
#   - free-standing 12+ digit i64 constants holding such hashes
#   - the [K x i8] length of a string constant that embeds one of those IDs, since the ID's digit
#     count varies
# Float constants (1.000000e+00), offsets and array sizes are deliberately NOT matched, and the
# general 6-digit threshold is not lowered: shorter runs of digits are real constants.
#
# Deliberately NOT masked, because they are not naming artefacts but hash-ordered choices:
#   - the order of union members inside generated names such as ___cast<!ts.union<...>,...> and
#     ___bin_op_plus<!ts.union<...>,...>
#   - union storage-type choice between equal-size members (findMaxSizeType breaks ties in
#     member order)
# A file hit by either is noisy by nature; compare it across two runs before calling a difference
# real. Known noisy files in tslang/test/tester/tests: typeGuardOfFormTypeOfBoolean,
# typeGuardFunction, 00funcs, 00union_bin_ops2, conditionalTypes2 (hash-ordered union choices),
# and - before FH<n> was masked - 00global_const_object_method and 00object_func.
sed -E 's/([A-Za-z_$]\.?)[0-9]{6,}/\1N/g; s/FH[0-9]+/FHN/g; s/\.[0-9]{6,}\.\.vtbl/.N..vtbl/g; s/(^|[^A-Za-z0-9_.$])-?[0-9]{12,}/\1N/g' "$1" \
  | sed -E '/c"[^"]*FHN/ s/\[[0-9]+ x i8\]/[K x i8]/g'
