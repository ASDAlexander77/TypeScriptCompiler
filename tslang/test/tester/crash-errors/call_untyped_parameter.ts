var x = {
    doStuff: (callback) => () => {
        return callback(this);
    }
};
