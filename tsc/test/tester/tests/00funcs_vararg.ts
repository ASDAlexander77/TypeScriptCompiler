function fn5(x: string, ...y: any[]) {
    print("count: ", y.length);
    assert(y.length == 3);
}

/*
function objectBindingPattern({foo}: any) {
    'use strict';
}

function arrayBindingPattern([foo]: any[]) {
    'use strict';
}
*/

function main() {
    fn5("hello", 1, 2, "str");
    print("done.");
}