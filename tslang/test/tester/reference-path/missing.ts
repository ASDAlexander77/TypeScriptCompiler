// A referenced file that is not there fails the compile. It used to print "can't open file",
// naming the main file, and compile the program anyway.
/// <reference path="no_such_file.d.ts" />

print("done.");
