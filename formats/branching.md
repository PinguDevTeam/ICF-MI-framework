# Branching and Pull Requests

This markdown file describes the structure we will follow for branches and how we will tackle the pull requests.

## Branches

For development, we will have a `dev` branch that has `main` as its parent branch. For issues labeled as `enhancement`, we will create branches out from `dev` using the naming convention that we will discuss below.

Note that we will *never* attempt to push any feature enhancements to `main` branch. All features will be unified and tested in `dev` branch before releasing them to `main`. Only `chore` type pull requests addressing issues in the general repository structure can be pushed directly from issue branches.

In the repository we will have the following subdirectories:
- __src__ : for source files
- __doc__ : for documentation
- __examples__ : for atomic examples
- __data__ : for dataset
- __formats__ : for guidelines
- __.github__ : for github templates

Each folder will be properly documented through `README.md` files that describe the contents of each subdirectory.

Here for this pipeline we have 3 major development items

1. Feature extraction
2. Surrogate modeling
3. Model identifiability

Each of those major development items will have individual subdirectories under `/src`.

## Issues

Each branch will have at least one related issue. You can have address more than one issue with a branch but please try to tackle a reasonable number of issues with a single branch. It's okay for you to have/work on more than one branch.

The naming convention for each branch will have the following structure

```<tag>/<your name>/<ID's issue(s) you are addressing>```

If you are addressing more than one issue with a single branch, combine the branch numbers with underscores. An example with mutliple branches is

```chore/doruk/1_2```

This means `doruk` is addressing the issues #1 and #2 that falls under `chore` category.

You will be using ```git checkout <branch name>``` command from command line to switch between branches (or you can use the branch button bottom left corner of VSCode). Please note that you have to either commit stash all your changes in one branch before checking out another branch.

To create a new branch please use

```git checkout -b <branch name> <name of the parent branch>```

command. For example, if `sahil` is going to work on a `bug` on `dev` branch that was reported in issue `#31`, he will use

```git checkout -b bug/sahil/31 dev```

command. Each issue will be referred in a pull request and will be linked to a milestone. So it's important that branches will follow a naming convention.


## Pull Requests (PRs)

Pull requests will follow a certain format as well. The name of each pull request will describe the essence of what changes are being proposed. Please refrain from using the branch name in the PR title since it does not convey a lot of information about the changes that have been made.

It's the responsibility of the requestor to assign reviewers (as if there are too many options there) and address/delegate the requests of the reviewers accordingly. Once all the requests are done, reviewers give the green light to the PR and then it's again the responsibility of the requestor to merge the branches to complete the PR. The branch related to the PR will be deleted if it does not have any children depending on it.
