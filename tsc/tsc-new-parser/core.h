#ifndef CORE_H
#define CORE_H

#include "config.h"
#include "undefined.h"

template <typename T, typename U>
auto copy(std::vector<T> &to_vector, const std::vector<U> &from_vector) -> void
{
    for (auto &item : from_vector)
    {
        to_vector.push_back(item);
    }
}

template <typename T, typename U>
auto copy(T &to_vector, const U &from_vector) -> void
{
    for (auto &item : from_vector)
    {
        to_vector.push_back(item);
    }
}

template <typename T, typename U>
auto forEach(T array, std::function<U(std::remove_reference_t<decltype(array[0])>, number)> callback = nullptr) -> U
{
    if (array.size())
    {
        for (auto i = 0; i < array.size(); i++)
        {
            auto result = callback(array[i], i);
            if (result)
            {
                return result;
            }
        }
    }

    return U();
}

template <typename T, typename U>
auto forEach(T array, std::function<U(std::remove_reference_t<decltype(array[0])>)> callback = nullptr) -> U
{
    if (array.size())
    {
        for (auto i = 0; i < array.size(); i++)
        {
            auto result = callback(array[i]);
            if (result)
            {
                return result;
            }
        }
    }

    return U();
}

template <typename T>
auto some(std::vector<T> array, std::function<boolean(T)> predicate = nullptr) -> boolean
{
    if (!array.empty())
    {
        if (predicate)
        {
            for (const v : array)
            {
                if (predicate(v))
                {
                    return true;
                }
            }
        }
        else
        {
            return array.size() > 0;
        }
    }
    return false;
}

template <typename T>
auto some(T array, std::function<boolean(decltype(array[0]))> predicate = nullptr) -> boolean
{
    if (array.size())
    {
        if (predicate)
        {
            for (auto &v : array)
            {
                if (predicate(v))
                {
                    return true;
                }
            }
        }
        else
        {
            return array.size() > 0;
        }
    }
    return false;
}

/** Works like Array.prototype.find, returning `undefined` if no element satisfying the predicate is found. */
template <typename T>
auto find(std::vector<T> array, std::function<boolean(T)> predicate) -> T {
    for (auto value : array) {
        if (predicate(value)) {
            return value;
        }
    }
    return undefined;
}

template <typename T>
auto find(std::vector<T> array, std::function<boolean(T, number)> predicate) -> T {
    for (auto i = 0; i < array.size(); i++) {
        auto value = array[i];
        if (predicate(value, i)) {
            return value;
        }
    }
    return undefined;
}

template <typename T>
auto find(T array, std::function<boolean(std::remove_reference_t<decltype(array[0])>)> predicate) -> std::remove_reference_t<decltype(array[0])> {
    for (auto value : array) {
        if (predicate(value)) {
            return value;
        }
    }
    return undefined;
}

template <typename T>
auto sameMap(T array, std::function<std::remove_reference_t<decltype(array[0])>(std::remove_reference_t<decltype(array[0])>)> f) -> T {
    if (array) {
        for (auto i = 0; i < array.size(); i++) {
            auto &item = array[i];
            auto mapped = f(item);
            if (item != mapped) {
                T result;
                copy(result, array);
                result.push_back(mapped);
                for (i++; i < array.size(); i++) {
                    result.push_back(f(array[i]));
                }
                return result;
            }
        }
    }
    return array;
}

template <typename T>
auto sameMapWithNumber(T array, std::function<std::remove_reference_t<decltype(array[0])>(std::remove_reference_t<decltype(array[0])>, number)> f) -> T {
    if (array) {
        for (auto i = 0; i < array.size(); i++) {
            auto &item = array[i];
            auto mapped = f(item, i);
            if (item != mapped) {
                T result;
                copy(result, array);
                result.push_back(mapped);
                for (i++; i < array.size(); i++) {
                    result.push_back(f(array[i], i));
                }
                return result;
            }
        }
    }
    return array;
}

template <typename T>
auto toOffset(T array, number offset) -> number
{
    return offset < 0 ? array.size() + offset : offset;
}

template <typename T, typename U>
auto addRange(T to, U from, number start = -1, number end = -1) -> T
{
    start = start == -1 ? 0 : toOffset(from, start);
    end = end == -1 ? from.size() : toOffset(from, end);
    for (auto i = start; i < end && i < from.size(); i++)
    {
        if (!!from[i])
        {
            to.push_back(from[i]);
        }
    }

    return to;
}

template <typename T>
auto findIndex(T array, std::function<boolean(decltype(array[0]), number)> predicate, number startIndex = 0) -> number
{
    for (auto i = startIndex; i < array.size(); i++)
    {
        if (predicate(array[i], i))
        {
            return i;
        }
    }
    return -1;
}

template <typename T>
auto lastOrUndefined(T array) -> std::remove_reference_t<decltype(array[0])>
{
    auto len = array.size();
    if (len > 0)
    {
        return array[len];
    }

    return undefined;
}

template <typename T>
auto arraysEqual(const std::vector<T> &a, const std::vector<T> &b) -> boolean
{
    if (a.size() != b.size())
    {
        return false;
    }

    auto i = 0;
    for (auto &ai : a)
    {
        if (ai != b[i++])
        {
            return false;
        }
    }

    return true;
}

template <typename T>
auto arraysEqual(T &a, T &b) -> boolean
{
    if (a.size() != b.size())
    {
        return false;
    }
        
    auto i = 0;
    for (auto &ai : a)
    {
        if (ai != b[i++])
        {
            return false;
        }
    }

    return true;
}


template <typename T>
auto compareComparableValues(T a, T b)
{
    return a == b ? Comparison::EqualTo : a < b ? Comparison::LessThan
                                                : Comparison::GreaterThan;
}

template <typename T>
auto compareValues(T a, T b) -> Comparison
{
    return compareComparableValues(a, b);
}

template <typename T>
using Comparer = std::function<Comparison(T, T)>;

template <typename T, typename U>
auto binarySearch(const std::vector<T> &array, T value, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0) -> number
{
    return binarySearchKey<T, U>(array, keySelector(value, -1), keySelector, keyComparer, offset);
}

template <typename T, typename U>
auto binarySearchKey(const std::vector<T> &array, U key, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0) -> number
{
    if (!array)
    {
        return -1;
    }

    auto low = offset;
    auto high = array.size() - 1;
    while (low <= high)
    {
        auto middle = low + ((high - low) >> 1);
        auto midKey = keySelector(array[middle], middle);
        switch (keyComparer(midKey, key))
        {
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

template <typename T, typename U>
inline auto append(T arr, U value) -> T
{
    arr.push_back(value);
    return arr;
}

// trim from start (in place)
inline static void ltrim(string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](char_t ch) {
                return !std::isspace(ch);
            }));
}

// trim from end (in place)
inline static void rtrim(string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](char_t ch) {
                return !std::isspace(ch);
            }).base(),
            s.end());
}

// trim from both ends (in place)
inline static auto trim(const string &s) -> string
{
    auto n = s;
    ltrim(n);
    rtrim(n);
    return n;
}

static string join(const std::vector<string> &v)
{
    string s;
    for (auto &p : v)
    {
        s += p;
    }

    return s;
}

#endif // CORE_H