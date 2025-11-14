# Instructions

Find and repair the bugs in `main()` in `main.cc`.

## Specifications

You are getting closer to the end!

You've implemented the matching part of the algorithm. You've implemented the
scoring part of the algorithm. Now, these pieces are coming together in a 
large file called `main.cc`.

For Phase I of `main.cc` (all that you will cover this week), you will patch in
the some of the remaining nuances of the algorithm.

### Explaining `main.cc`

Initially, there are some configs for you to change. Keep them as is.

There are some helper functions. The most important of which is the
`create_user_from_json` which literally creates a new `User` object from the 
parsed JSON data. You likely won't need it, but if you intend to add onto it
in the future, it would be important.

Finally, you get to `main()`. What you're looking at is a slimmed down version
of `main()` from the true algorithm. The real algorithm has an additional 400
lines of code, but you probably don't need them to understand what the `main`
function is doing.

There is some setup for reading in the args, and then it transitions to
running the algorithm per school (Remember, Datamatch is not just for Harvard
students!). 

We first process the answer choice cosine similarties (based off a Python script
that we did not give you). We create all the users at this specific college. We
then find the distribution of answers per question.

Next, we determine the specific weights and kinds of matches per pair `(i, j)`.
Then, compute the scores per pair, and match! Then, it's as simple as outputting
it to some JSON and freeing any memory.

This explanation handwaves a lot of the current bugs with the implementation.

We are only asking you to fix two, as per the below.

### Dorm Matches

Currently, we read "noDormMatch" but don't do anything with it. If `true`, it's
supposed to prevent matching with people in the same dorm. Implement this. You
may find modifying the values of the `matches` and `matchTypes` matrices useful.

### Reporting your own bug

> Run your tests to check your work now. Save this step for last. Only proceed when you feel you have completed the rest. Make sure you've committed and pushed your changes.

Part of working with a team on such a big project is spotting and fixing bugs. 
But, we also like to keep track of possible planned enhancements, or 
suggestions for documentation changes, etc. You can call these problems 
"issues."

We prefer using Github Issues to manage our project's issues, since it's built 
right into the version control system we use.

#### Creating an Issue

For this part, you will find your own bug in the code (or perhaps, if you can't, 
find one, consider a possible enhancement to make to our algorithm). 
On your forked repo's Github page, go to the Issues tab and select New Issue.

Create a subject title for your issue, and leave a description good enough so 
that anyone who skims your code can have a somewhat detailed understanding of 
the bug or enhancement. Sometimes, for bugs, its useful to showcase the steps 
to reproduce.

You'll notice some features on the right hand side of the screen, such as 
"Assignees," "Labels," "Projects," and "Milestone." Assign yourself this 
issue (so it's clear you're going to take up the task of working on it) 
and give it a proper label. You can always change these later.

Create your issue.

#### Creating a branch

Afterwards, you'll notice that there is a number next to the title of your 
issue. This is our issue number.

Go to Github Desktop and create a new branch. If your issue is a bugfix, title 
it `fix/issue-<issue number>-<optional 1-3 word description>`. If your issue is 
an feature request, title it `feat/issue-<issue number>-<optional 1-3 word description>`.

For instance, if my bug was about fixing an off by zero error in a `for` loop,
you might make a bug called `fix/issue-123`, since the change is so small.
Alternatively, if you were to fix a bug surrounding dorm matches, you might
call it `fix/issue-123-dorm-match`.

#### Creating a pull request

All of your work now lives on this branch. Go ahead and fix your bug or make
your enhancement.

When finished, commit and push your changes. Then, preview your pull request and
then create a pull request on the website. Make sure you're merging into your
`main` branch, and not `ivanharvard`'s `main` branch.

Give it a descriptive title of what happened after the changes were made, 
perhaps something like "Dorm matches correctly function." Now, in the 
description of the pull request, you can write certain keywords to let Github
know this pull request is related to an issue.

If, at the first line of the description you specifically say "Closes #123." 
then Github will close issue #123 if your pull request gets successfully 
merged. Finish creating your pull request.

Normally, this is where you'd stop, and in the real world, I would go in, review
your changes. You will simulate this by simply merging and closing the 
pull request, which will subsequently close the related issue.

Once you're done, your changes should live on the `main` branch. Switch over to
it and check the history to confirm this. Success!

# Checking your work

You can check if your cross-matching implementation works below. Depending on
the scale of your bugfix in the second part of this assignment, these checks
may no longer apply to you once you make those changes.

To compile, run
```bash
make main
./main data/FIREBASE_EXPORT.json data/sims/
```

To test your work using the autograder, run
```bash
check50 ivanharvard/datamatch-algo-comp/main/comp3 --local
```