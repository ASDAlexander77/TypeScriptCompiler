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

#endif // CORE_H