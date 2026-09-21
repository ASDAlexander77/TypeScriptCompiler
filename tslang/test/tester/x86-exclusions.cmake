# The tests whose 32-bit (i686) twin cannot run, read when TSLANG_TEST_X86 is on.
#
# Each entry is the x64 test name (test-..., not test-x86-...) and needs a one-line reason in
# TSLANG_X86_EXCLUDED_REASON_<name>. Its twin is still registered, with DISABLED, so ctest
# reports it as "Not Run (Disabled)" instead of leaving it out. Only a test that can never
# work at 32 bits belongs here; a 32-bit defect in the compiler is fixed, not listed.

set(TSLANG_X86_EXCLUDED
    test-compile-internals
    test-compile-rc-corpus-internals
    test-compile-none-corpus-internals
    )

# internals.ts, under each model: its first assert, on `inline_asm<i64>("xor $0, $0", "=r,r", v1)`,
# fails at i686 (0xC0000409 before the first print).
set(TSLANG_X86_EXCLUDED_REASON_test-compile-internals
    "`inline_asm<i64>` with an `=r` constraint: no single i686 register holds 64 bits")
set(TSLANG_X86_EXCLUDED_REASON_test-compile-rc-corpus-internals
    "`inline_asm<i64>` with an `=r` constraint: no single i686 register holds 64 bits")
set(TSLANG_X86_EXCLUDED_REASON_test-compile-none-corpus-internals
    "`inline_asm<i64>` with an `=r` constraint: no single i686 register holds 64 bits")
