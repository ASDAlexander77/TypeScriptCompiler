function main() {
    print(void 2 == "2"); // (void 2) == '2', returns false
    print(void (2 == "2")); // void (2 == '2'), returns undefined

    print("done.");
}
