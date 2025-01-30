// @strict-null false

type tm = [tm_sec: i32, tm_min: i32, tm_hour: i32, tm_mday: i32, tm_mon: i32, tm_year: i32, tm_wday: i32, tm_yday: i32, tm_isdst: i32];
const r: Reference<tm> = null;

let a = false;

if (r == null)
    a = true;

assert(a);

print("done.");