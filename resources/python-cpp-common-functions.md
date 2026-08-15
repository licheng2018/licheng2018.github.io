# Python and C++ Common Functions

Quick reference for common data structures and utility functions used in coding interviews.

## Contents

| Language | Topics |
|---|---|
| Python | [Queue](#queue) · [Set](#set) · [Stack](#stack) · [Hash Map / Dictionary](#hash-map--dictionary) · [Heap](#heap) · [String and Sorting](#string-and-sorting) · [String Utilities](#string-utilities) · [Common Constants and 2D Arrays](#common-constants-and-2d-arrays) |
| C++ | [Vector / 2D List](#vector--2d-list) · [Sort](#sort) · [Heap / Priority Queue](#heap--priority-queue) · [Set](#set-1) · [String Utilities](#string-utilities) · [Queue / Deque](#queue--deque) · [Stack](#stack-1) · [Pair](#pair) · [Common Constants and Conversions](#common-constants-and-conversions) |

## Python

### Queue

```python
from collections import deque

queue = deque([element])
queue.append(element)      # push
item = queue.popleft()     # pop from front
```

### Set

```python
hash_set = set([1, 2, 3])
hash_set.add(x)        # insert
hash_set.remove(x)     # remove, raises KeyError if x is missing
hash_set.discard(x)    # remove safely if x may be missing
```

### Stack

```python
stack = []
top = stack[-1]
item = stack.pop()
stack.append(char)
```

### Hash Map / Dictionary

```python
key2val = {}

if key in key2val:
    pass

key2val[key] = value
del key2val[key]
```

### Heap

```python
import heapq

heap = []
heapq.heappush(heap, element)
smallest = heapq.heappop(heap)
```

### String and Sorting

```python
s = "abc"

joined = " ".join(words)
hashable_key = tuple(s)
sorted_num = sorted(nums)
```

### String Utilities

```python
s = "abc"
c = "A"

ok1 = c.isalnum()
ok2 = c.isalpha()
ok3 = c.isdigit()
lower = c.lower()
upper = c.upper()

lower_s = s.lower()
upper_s = s.upper()

reversed_s = s[::-1]
s = s + "xyz"

# If you need mutable append-style updates:
chars = list(s)
chars.append("z")
s = "".join(chars)
```

### Common Constants and 2D Arrays

```python
max_num = float("inf")
min_num = float("-inf")

visited = [[0 for _ in range(n)] for _ in range(m)]
```

## C++

### Vector / 2D List

```cpp
#include <vector>
#include <string>
using namespace std;

vector<vector<string>> strlist;
strlist.push_back(row);
strlist.pop_back();

vector<vector<int>> visited(m, vector<int>(n, 0));
```

### Sort

```cpp
#include <algorithm>

sort(intervals.begin(), intervals.end());
```

### Heap / Priority Queue

```cpp
#include <queue>

priority_queue<pair<int, int>> heap;  // max heap by default

heap.push({a, b});
auto top_item = heap.top();
heap.pop();
```

For a min heap:

```cpp
priority_queue<
    pair<int, int>,
    vector<pair<int, int>>,
    greater<pair<int, int>>
> min_heap;
```

### Set

```cpp
#include <unordered_set>

unordered_set<int> num_set;
num_set.insert(num);
num_set.erase(num);
```

### String Utilities

```cpp
#include <string>
#include <algorithm>
#include <cctype>

string s = "abc";

bool ok1 = isalnum(c);
bool ok2 = isalpha(c);
bool ok3 = isdigit(c);
char lower = tolower(c);
char upper = toupper(c);

reverse(s.begin(), s.end());
s.append("xyz");

transform(s.begin(), s.end(), s.begin(), ::tolower);
transform(s.begin(), s.end(), s.begin(), ::toupper);
```

### Queue / Deque

```cpp
#include <deque>

deque<pair<int, int>> queue;

queue.push_back({i, j});
pair<int, int> node = queue.front();
queue.pop_front();
```

### Stack

```cpp
#include <stack>

stack<char> stk;

char top_char = stk.top();
stk.pop();
stk.push(char_value);
```

### Pair

```cpp
pair<int, int> a;

int x = a.first;
int y = a.second;
```

### Common Constants and Conversions

```cpp
#include <climits>
#include <string>

int max_number = INT_MAX;

int value = stoi(s);
string text = to_string(value);
```
