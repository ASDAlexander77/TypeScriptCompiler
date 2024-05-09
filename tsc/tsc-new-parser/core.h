#ifndef CORE_H
#define CORE_H

#include "config.h"
#include "undefined.h"

#include <algorithm>
#include <cmath>

namespace ts
{

template <typename T, typename U> auto copy(std::vector<T> &to_vector, const std::vector<U> &from_vector) -> void
{
    for (auto &item : from_vector)
    {
        to_vector.push_back(item);
    }
}

template <typename T, typename U> auto copy(T &to_vector, const U &from_vector) -> void
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

template <typename T> auto some(std::vector<T> array, std::function<boolean(T)> predicate = nullptr) -> boolean
{
    if (!array.empty())
    {
        if (predicate)
        {
            for (auto v : array)
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

template <typename T> auto some(T array, std::function<boolean(decltype(array[0]))> predicate = nullptr) -> boolean
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
template <typename T> auto find(std::vector<T> array, std::function<boolean(T)> predicate) -> T
{
    for (auto value : array)
    {
        if (predicate(value))
        {
            return value;
        }
    }
    return undefined;
}

template <typename T> auto find(std::vector<T> array, std::function<boolean(T, number)> predicate) -> T
{
    for (auto i = 0; i < array.size(); i++)
    {
        auto value = array[i];
        if (predicate(value, i))
        {
            return value;
        }
    }
    return undefined;
}

template <typename T>
auto find(T array, std::function<boolean(std::remove_reference_t<decltype(array[0])>)> predicate)
    -> std::remove_reference_t<decltype(array[0])>
{
    for (auto value : array)
    {
        if (predicate(value))
        {
            return value;
        }
    }
    return undefined;
}

template <typename T>
auto sameMap(T array, std::function<std::remove_reference_t<decltype(array[0])>(std::remove_reference_t<decltype(array[0])>)> f) -> T
{
    if (array)
    {
        for (auto i = 0; i < array.size(); i++)
        {
            auto &item = array[i];
            auto mapped = f(item);
            if (item != mapped)
            {
                T result;
                copy(result, array);
                result.push_back(mapped);
                for (i++; i < array.size(); i++)
                {
                    result.push_back(f(array[i]));
                }
                return result;
            }
        }
    }
    return array;
}

template <typename T>
auto sameMapWithNumber(T array,
                       std::function<std::remove_reference_t<decltype(array[0])>(std::remove_reference_t<decltype(array[0])>, number)> f)
    -> T
{
    if (array)
    {
        for (auto i = 0; i < array.size(); i++)
        {
            auto &item = array[i];
            auto mapped = f(item, i);
            if (item != mapped)
            {
                T result;
                copy(result, array);
                result.push_back(mapped);
                for (i++; i < array.size(); i++)
                {
                    result.push_back(f(array[i], i));
                }
                return result;
            }
        }
    }
    return array;
}

template <typename T> auto toOffset(T array, number offset) -> number
{
    return offset < 0 ? array.size() + offset : offset;
}

template <typename T, typename U> auto addRange(T to, U from, number start = -1, number end = -1) -> T
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

template <typename T> auto findIndex(T array, std::function<boolean(decltype(array[0]), number)> predicate, number startIndex = 0) -> number
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

template <typename T> auto firstOrUndefined(T array) -> std::remove_reference_t<decltype(array[0])>
{
    auto len = array.size();
    if (len > 0)
    {
        return array[1];
    }

    return undefined;
}

template <typename T> auto lastOrUndefined(T array) -> std::remove_reference_t<decltype(array[0])>
{
    auto len = array.size();
    if (len > 0)
    {
        return array[len - 1];
    }

    return undefined;
}

template <typename T> auto arraysEqual(const std::vector<T> &a, const std::vector<T> &b) -> boolean
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

template <typename T> auto arraysEqual(T &a, T &b) -> boolean
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

template <typename T> auto compareComparableValues(T a, T b)
{
    return a == b ? Comparison::EqualTo : a < b ? Comparison::LessThan : Comparison::GreaterThan;
}

template <typename T> auto compareValues(T a, T b) -> Comparison
{
    return compareComparableValues(a, b);
}

template <typename T> using Comparer = std::function<Comparison(T, T)>;

template <typename T, typename U>
auto binarySearchKey(const std::vector<T> &array, U key, std::function<U(T, number)> keySelector, Comparer<U> keyComparer,
                     number offset = 0) -> number
{
    if (array.size() <= 0)
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
auto binarySearch(const std::vector<T> &array, T value, std::function<U(T, number)> keySelector, Comparer<U> keyComparer, number offset = 0)
    -> number
{
    return binarySearchKey<T, U>(array, keySelector(value, -1), keySelector, keyComparer, offset);
}

template <typename T, typename U> inline auto append(T arr, U value) -> T
{
    arr.push_back(value);
    return arr;
}

// trim from start (in place)
inline static void ltrim(string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](char_t ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
inline static void rtrim(string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](char_t ch) { return !std::isspace(ch); }).base(), s.end());
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

/**
 * Given a name and a list of names that are *not* equal to the name, return a spelling suggestion if there is one that is close enough.
 * Names less than length 3 only check for case-insensitive equality.
 *
 * find the candidate with the smallest Levenshtein distance,
 *    except for candidates:
 *      * With no name
 *      * Whose length differs from the target name by more than 0.34 of the length of the name.
 *      * Whose levenshtein distance is more than 0.4 of the length of the name
 *        (0.4 allows 1 substitution/transposition for every 5 characters,
 *         and 1 insertion/deletion at 3 characters)
 *
 * @internal
 */

static string str_tolower(string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

static string str_toupper(string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::toupper(c); });
    return s;
}


using fnumber = double;
static auto levenshteinWithMax(string s1, string s2, number max) -> number {
    std::vector<fnumber> previous(s2.size() + 1);
    std::vector<fnumber> current(s2.size() + 1);
    /** Represents any value > max. We don't care about the particular value. */
    auto big = max + 0.01;

    for (auto i = 0; i <= s2.size(); i++) {
        previous[i] = i;
    }

    for (auto i = 1; i <= s1.size(); i++) {
        auto c1 = s1[i - 1];
        auto minJ = (fnumber) std::ceil(i > max ? i - max : 1);
        auto maxJ = (fnumber) std::floor(s2.size() > max + i ? max + i : s2.size());
        current[0] = i;
        /** Smallest value of the matrix in the ith column. */
        auto colMin = i;
        for (auto j = 1; j < minJ; j++) {
            current[j] = big;
        }
        for (auto j = minJ; j <= maxJ; j++) {
            // case difference should be significantly cheaper than other differences
            auto substitutionDistance = std::tolower(s1[i - 1]) == std::tolower(s2[j - 1])
                ? (previous[j - 1] + 0.1)
                : (previous[j - 1] + 2);
            auto dist = c1 == s2[j - 1]
                ? previous[j - 1]
                : std::min(std::min(/*delete*/ previous[j] + 1, /*insert*/ current[j - 1] + 1), /*substitute*/ substitutionDistance);
            current[j] = dist;
            colMin = std::min((fnumber)colMin, dist);
        }
        for (auto j = maxJ + 1; j <= s2.size(); j++) {
            current[j] = big;
        }
        if (colMin > max) {
            // Give up -- everything in this column is > max and it can't get better in future columns.
            return undefined;
        }

        auto temp = previous;
        previous = current;
        current = temp;
    }

    auto res = previous[s2.size()];
    return res > max ? undefined : res;
}

template <typename T>
auto getSpellingSuggestion(string name, std::vector<T> candidates, std::function<string(T)> getName) -> T {
    auto maximumLengthDifference = std::max(2.0, std::floor(name.length() * 0.34));
    auto bestDistance = std::floor(name.length() * 0.4) + 1; // If the best result is worse than this, don't bother.
    T bestCandidate;
    for (auto candidate : candidates) {
        auto candidateName = getName(candidate);
        if (!candidateName.empty() && std::abs((const long)(candidateName.size() - name.length())) <= maximumLengthDifference) {
            if (candidateName == name) {
                continue;
            }
            // Only consider candidates less than 3 characters long when they differ by case.
            // Otherwise, don't bother, since a user would usually notice differences of a 2-character name.
            if (candidateName.length() < 3 && str_tolower(candidateName) != str_tolower(name)) {
                continue;
            }

            auto distance = levenshteinWithMax(name, candidateName, bestDistance - 0.1);
            if (distance == undefined) {
                continue;
            }

            // TODO: finish it
            //Debug::_assert(distance < bestDistance); // Else `levenshteinWithMax` should return undefined
            bestDistance = distance;
            bestCandidate = candidate;
        }
    }
    return bestCandidate;
}

} // namespace ts

#endif // CORE_H