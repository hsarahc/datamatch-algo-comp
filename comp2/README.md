# Instructions

Complete `check_compatibility` and `bonus` in `scoring.cc`.

## Specifications

In `matching.cc`, you saw how to implement certain C-style functions. In
`scoring.cc`, you will learn about managing classes and their attributes in C++.
For instance, you may be given a `User` object called `u1` defined as so:
```cpp
User u1;
```
Looking at the `user.hpp` file, you may notice that being a `User` gives you
certain preset (and unset) values. Let's access one of those attributes:
```cpp
std::cout << u1.seriousness << std::endl;
```
This prints out the seriousness of this user. For now, this is 0. We can update
it as so:
```cpp
u1.seriousness = 10;
std::cout << u1.seriousness << std::endl;
```
This code block will now print out `10`. 

Another thing to look at is this code piece that you may see lying around:
```cpp
std::vector<int>
```
We can break this down into parts. `std` refers to the standard library namespace.
In other words, C++ has provided us with generic, basic, built-in data structures
and methods (so long as we include their header files), and labeled them as 
belonging to some `std` library. One example you have already seen is 
`cout` and `endl`, but now we are introducing more. 

`vector` is a data type under the `std` library. You can import it into your
code by having the header
```cpp
#include <vector>
```
at the top of your file. You may be familiar with Python's lists. This is 
essentially a Python list, except with the caveat that all elements within the
vector must be of the same type, and in this case, we've made sure that the 
vector can only hold `int`s.

All this to say, we're creating a dynamic array that can only hold `int` values.

Being a `std::vector` comes with certain methods built in. For instance, the
method `.size()` gives you the length of the vector, and `.push_back()` appends
a value to the end of the vector.
```cpp
std::vector<int> v;
v.push_back(1)
v.push_back(2)
std::cout << v.size() << std::endl; // prints 2
```
The equivalent in Python might look like:
```python
v: list[int] = []
v.append(1)
v.append(2)
print(len(v))
```

You will not have to use very many things from the `std` namespace to complete
this comp assignment, but we will introduce more of them as time continues. 
However, you'll likely have to modify or reference the `user.hpp` file to
complete it, and so our goal is that you will get comfortable with classes 
during this assignment!

### `check_compatibility`

`check_compatibility` is our function that determines if two `User`s are
compatible. Implement the below rules for `User u1` and `User u2` inside
of this function.

Rules:
- Datamatch users can't match with themselves.
- Normally, Datamatch allows for cross-school dates for certain colleges.
For instance, MIT and Harvard students can go on dates if the user prefers.
For this function, assume that this feature has been disabled. If `u1` and `u2`
attend different colleges, they are not compatible.
- Datamatch allows people to seek dates for love, or dates for friendship.
`u1` and `u2` can't be compatible if `u1` is seeking love, and `u2` is seeking 
friendship, and vice versa. `get_match_type()` may be useful here.
- A few years ago, Datamatch implemented a crush roulette system that allows a 
person to "toss a coin into a well" to "increase the chances of being matched."
If `u1` and `u2` are each other's crushes, then no matter what, they are 
compatible.
- To avoid seniors getting matched with freshman, we make sure that two users
aren't matched if their class year difference is too great. `u1` and `u2` cannot
be farther than `YEAR_DIFF_MAX` in class years to be compatible.
- Some users prefer people around their age (or older, or younger). Enforce the
class year requirements first, then make sure that `u1` and `u2`'s age
preferences (if they exist) align with `u2` and `u1`'s ages, respectively, and
vice versa.
- Some users prefer not to date members within their house. Enforce this rule.
- Some users have blocked matching with anyone from certain houses entirely.
Enforce this rule.

### `bonus`

As is the norm with Algo team, you will often be tasked with implementing a new
feature for the Datamatch algorithm to take into consideration. Our code 
currently looks at stuff like bio similarity, spotify artist similarity, etc.

Either by modifying the `user.hpp` file or referencing it, design a new, 
additional weight to the final score. 

Example:
The crush roulette currently only considers mutual crushes. Let's say that if
the crush is actually one-way (and they are compatible), then we return some `x` 
amount of points.

Note:
If you modify the `user.hpp` file, do not modify existing variables.

# Checking your work

You can check whether your code compiles by running:
```bash
make scoring
```
If no error messages appear, the code compiled successfully.

You can check an individual test file by running
```bash
make <test number>
```

You can check what we expect your file to output by running
```bash
make answers-<test number>
```

You can check all of the tests, and some useful output, by running
```bash
check50 ivanharvard/datamatch-algo-comp/main/comp2 --local
```

