# Running the tests on Android

[run_tests.mjs](run_tests.mjs) builds the tester's tests and the default library's tests for
Android, and runs them on a device or an emulator through adb. Which tests there are, and when one
passes, is shared with the wasm tests: [../suites.mjs](../suites.mjs).

    node run_tests.mjs [--build | --run] [--suite=tester,defaultlib] [--mm=gc,rc,none,own] [--abi=x86_64] [-j<n>] [filter]

- `--build` compiles every test with `-mtriple=<abi's triple, API 29> --opt` and links it through
  the NDK (`ANDROID_NDK_HOME` or `--android-ndk-path`). It uses the Android default library, the
  collector and the async runtime: `scripts\build_gc_release_android.bat`,
  `scripts\build_tslang_runtime_release_android.bat` and the default library's
  `scripts\build_android.bat`, or a release package's (`--android-lib=<package>\android\<abi>\lib`).
  Each program is stripped. The programs and `manifest.json` go to `out/`.
- `--run` needs only adb and `out/`. It runs one leg (a suite under a model) at a time in
  `/data/local/tmp/tslang-tests`: it pushes the leg's programs, runs them on the device four at a
  time, each under `timeout` and a memory limit, and pulls back each one's exit code and the last
  lines of its output. The summary and every failure are written to `out/results.txt`.
- With neither flag it does both.

The tests in [android_skips.txt](android_skips.txt) are skipped, each with its reason. So are the
default library's tests marked `// gc only` (except under gc) or `// not own` (under own). Under
`-mm=own`, a test that does not compile for the host either is listed as NATIVE, as in the wasm
runner.

## An emulator

An x86_64 system image runs the x86_64 programs natively. For example, with the Android SDK:

    sdkmanager "system-images;android-29;default;x86_64" "emulator" "platform-tools"
    avdmanager create avd -n tslang_api29_x86_64 -k "system-images;android-29;default;x86_64"
    emulator -avd tslang_api29_x86_64 -no-window -no-audio -no-boot-anim -no-snapshot -memory 2048

arm64-v8a and armeabi-v7a programs need an arm device: an x86_64 emulator runs them only through
its binary translator, which is not reliable enough to test with.

## In CI

[test-android.yml](../../../.github/workflows/test-android.yml) builds the x86_64 programs on
Windows, with a Windows package's compiler and Android libraries (the only package that ships
them), and runs them on an API 29 x86_64 emulator on Linux. A release runs it against the package
it has just built (`test_android` in `create-release.yml`); a pull request that changes the runner
runs it against the latest published release.
