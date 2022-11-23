# Commits
Commits are a very important part of the development process that helps documenting intermediate steps and allows backtracking
There are some practices we need to follow (or stay away) for a better life

1. **_DON'T_** stage all your changes in a single commit: Commits should be atomic and very specific. This will allow easier backtracking if things go south after a commit.
2. Be specific in your commit message: This will allow us to follow the changes even without looking at the changed code.
    _How can I know if I've staged too much?_: If you can't sum up what you have staged in a couple words, then you have staged too much.
3. _Always_ give details in the next line: Give (at least 1 sentence long) summary in the second line. This will help us understand what you have done if the first line is not enough.
4. _Follow_ the format: Having a unified commit message format will allow us to search efficiently.
    We will follow the format:

```
(<tag>) <subject>
<Explanation in full sentences>
```

_tag_: Pick the most appropriate tag from the list below
_subject_: Brief summary(30-40 characters) of the changes with couple words. Use imperative mood. Don't end with a period.

## Tags for commit messages

- feat – If you work on a new feature
- fix – If your code changes help in fixing some bug
- docs – Documentation related changes
- style – Changes related to formatting
- refactor – If changes are related to just restructuring the codebase
- test – If new tests are added or old tests are modified
- chore – Maintenance changes for having a repository
- WIP – If you need to change computers while working and didn't get the chance to finish your work but need to save it for now
- CI – If you are merging any changes from a parent branch (stands for Continuous Integration)

# Proper commit messages (from #4)
This is what I'm expecting in this repository:

```
(chore) create todo ticket
Created to-do item ticket template for communication.
```
This is a 7/10 commit message:

```
(chore) create bug report template
Created markdown file for bug report issues.
```

Note that the scope does not exactly indicate to which file I've modified. You can still understand it from the subject line, hence 7/10.

Now, what **_not_** to do:
```
updated commits.md
```
Commit message not descriptive at all. Only states which file is modified, no mention of _what_ is done. No detailed explanation given. 0% backtracking possibility, you will 100% end up checking the changed code to understand what happened. Don't do it. Just.. Don't.

If I see you doing this, I'll hunt you down.
