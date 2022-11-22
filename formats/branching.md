# Branching and Pull Requests

This markdown file describes the structure we will follow for branches and how we will tackle the pull requests.

## Branches

Here for this pipeline we have 4 major development items

1. Dataset
2. Feature extraction
3. Surrogate modeling
4. Model identifiability

Each of those major development items will have individual branches. We will be working with branches *under* those major development branches.

Each branch will have at least one related issue. You can have address more than one issue with a branch but please try to tackle a reasonable number of issues with a single branch. It's okay for you to have/work on more than one branch.

The naming convention for each branch will have the following structure

```<tag>/<your name>/<number of issue(s) you are addressing>```

If you are addressing more than one issue with a single branch, combine the branch numbers with underscores. An example with mutliple branches is

```chore/doruk/1_2```

This means `doruk` is addressing the issues #1 and #2 that falls under `chore` category.

You will be using ```git checkout <branch name>``` command from command line to switch between branches (or you can use the branch button bottom left corner of VSCode). Please note that you have to either commit stash all your changes in one branch before checking out another branch.

To create a new branch please use

```git checkout -b <branch name> <name of the branch that you are branching out from>```

command. For example, if `sahil` is going to work on a `bug` on `GIM` branch that was reported in issue `#31`, he will use

```git checkout -b bug/sahil/31 GIM```

command. Each issue will be referred in a pull request and will be linked to a milestone. So it's important that branches will follow a naming convention.


## Pull Requests (PRs)

Pull requests will follow a certain format as well. The name of each pull request will describe the essence of what changes are being proposed. Please refrain from using the branch name in the PR title since it does not convey a lot of information about the changes that have been made.

It's the responsibility of the requestor to assign reviewers (as if there are too many options there) and address/delegate the requests of the reviewers accordingly. Once all the requests are done, reviewers give the green light to the PR and then it's again the responsibility of the requestor to merge the branches to complete the PR. The branch related to the PR will be deleted if it does not have any children depending on it.