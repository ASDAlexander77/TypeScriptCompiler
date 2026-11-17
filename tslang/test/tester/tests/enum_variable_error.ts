// TSLANG_ISSUES 103: a variable can't hold an enum itself, which is its members at compile time with
// nothing at run time (it hung the compiler); a `const` naming it works
enum Days {
    Monday,
    Tuesday
}

const days = Days;
var daysValue = Days;

function main() {
    print(days.Tuesday);
    print("done.");
}
