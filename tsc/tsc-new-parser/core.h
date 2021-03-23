#ifndef CORE_H
#define CORE_H

#include "config.h"
#include "undefined.h"

template <typename T, typename U>
auto forEach(std::vector<T> array, std::function<U(T, number)> callback = nullptr) -> U {
    if (!array.empty()) {
        for (let i = 0; i < array.size(); i++) {
            auto result = callback(array[i], i);
            if (result) {
                return result;
            }
        }
    }

    return U();
}

template <typename T>
auto some(std::vector<T> array, std::function<boolean(T)> predicate = nullptr) -> boolean {
    if (!array.empty()) {
        if (predicate) {
            for (const v : array) {
                if (predicate(v)) {
                    return true;
                }
            }
        }
        else {
            return array.size() > 0;
        }
    }
    return false;
}

template <typename T>
auto some(T array, std::function<boolean(decltype(array[0]))> predicate = nullptr) -> boolean {
    if (!array.empty()) {
        if (predicate) {
            for (const v : array) {
                if (predicate(v)) {
                    return true;
                }
            }
        }
        else {
            return array.size() > 0;
        }
    }
    return false;
}

template <typename T>
auto toOffset(T array, number offset) -> number {
    return offset < 0 ? array.size() + offset : offset;
}

template <typename T, typename U>
auto addRange(T to, U from, number start = -1, number end = -1) -> T {
    start = start == -1 ? 0 : toOffset(from, start);
    end = end == -1 ? from.size() : toOffset(from, end);
    for (auto i = start; i < end && i < from.size(); i++) {
        if (from[i] != undefined) {
            to.push_back(from[i]);
        }
    }

    return to;
}

template <typename T>
auto findIndex(T array, std::function<boolean(decltype(array[0]), number)> predicate, number startIndex = 0) -> number {
    for (auto i = startIndex; i < array.size(); i++) {
        if (predicate(array[i], i)) {
            return i;
        }
    }
    return -1;
}

template <typename T>
auto lastOrUndefined(T array) -> decltype(array[0]) {
    return array.size() == 0 ? decltype(array[0])(undefined)/*undefined*/ : array[array.size() - 1];
}


template <typename T>
auto arraysEqual(const std::vector<T> &a, const std::vector<T> &b) -> boolean {
    return std::equal(a.begin(), a.end(), b.begin());
}

template <typename T>
auto compareComparableValues(T a, T b) {
    return a == b ? Comparison::EqualTo :
        a < b ? Comparison::LessThan :
        Comparison::GreaterThan;
}

template <typename T>
auto compareValues(T a, T b) -> Comparison {
    return compareComparableValues(a, b);
}

template <typename T>
using Comparer = std::function<Comparison(T, T)>;

template <typename T, typename U>
auto binarySearch(const std::vector<T> &array, T value, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0) -> number {
    return binarySearchKey<T, U>(array, keySelector(value, -1), keySelector, keyComparer, offset);
}

template <typename T, typename U>
auto binarySearchKey(const std::vector<T> &array, U key, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0) -> number {
    if (!array) {
        return -1;
    }

    auto low = offset;
    auto high = array.size() - 1;
    while (low <= high) {
        auto middle = low + ((high - low) >> 1);
        auto midKey = keySelector(array[middle], middle);
        switch (keyComparer(midKey, key)) {
            case Comparison::LessThan:
                low = middle + 1;
                break;
            case Comparison::EqualTo:
                return middle;
            case Comparison::GreaterThan:
                high = middle - 1;
                break;
        }
    }

    return ~low;
}

#endif // CORE_H