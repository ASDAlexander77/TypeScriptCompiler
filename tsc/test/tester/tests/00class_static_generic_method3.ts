class CustomDate {
    static parse(dateString) {
        return this.createDate(1, 2, 3);
    }

    // Helper function to create a Date object using UTC time
    static createDate(year, month, day, hour = 0, minute = 0, second = 0) {
        return year + month;
    }
}

CustomDate.parse("test");

print("done.");

