// -mm=own rejects: `getarr` returns the value of a variable it captures, which the enclosing
// function still owns.
function main() {
    let arr = [1];
    function getarr() {
        return arr;
    }
    getarr()[0]++;
    print(arr[0]);
}
