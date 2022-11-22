# Workflow

This document explains the workflow we will follow as a development group and provides examples for git commands we will use.

## Creating an issue

The workflow starts with creating an issue on github.

You can create issues for the following 3 cases:

1. **[BUG]** : reporting a bug
2. **[FR]** : requesting a feature from a collaborator
3. **[TODO]** : creating a to-do item for workflow transparency and milestone tracking 

There are templates for all of the cases. You just need to follow the directions in each template. Don't forget to assign appropriate labels to the issue, and assign the issue to the relevant person and milestone. 

## Creating a branch

Once you know which issue(s) you will address you will need to create a branch. Please check the details in the [branching guideline](https://github.com/PinguDevTeam/ICF-MI-framework/formats/branching.md) for further details.

You can create a branch using the following command

`git checkout -b <branch name> <name of the parent branch>`

Alternatively you can create branches using

`git branch <branch name>`

However, you need to be careful with that. Using `git branch` creates branch from your currently checked-out branch. Similar to the `git checkout` procedure, if you specify the parent branch in the command as

`git branch <branch name> <name of the parent branch>`

you can specify from which branch you want to branch out.

An important topic here is the distinction between `main` and `origin/main`. `main` is the **clone** of the `main` branch on the remote repository (in our case github), whereas `origin/main` is the `main` branch of github. The same applies for any branch that is on github. So please keep the difference between those two in mind while creating branches for your issues.

If you want to delete a branch you can use the command

`git branch -d <name of the branch>`

## Working on a branch

Each time you start to work on a branch, please **fetch** all the changes from the reposit using `git fetch` and check if there are any changes affecting your current active branch using 

`git status`

If you don't want to see too much detail on the status of your current branch and just want to see which files are modified/created, you can use the command 

`git status -s`

If there are changes affecting your active branch. Please get all the changes from the relevant parent branch using the command

```git merge <name of child branch> origin/<path of the parent branch>```

Please commit those merges appropriately according to the naming convention. 

Once you work on a small task towards addressing a issue, please **commit** your changes in small packages. To commit your changes, you first need to **add/stage** your changes. In VSCode you can see the changes you made from the color on the left of your code. **Blue** indicates a modification, **Green** indicates new lines, and **Red** indicates deletion. On VSCode you can `stage` your changes by clicking on the colored strip and clicking on the `+` sign on the top right corner of the opened box showing the content of the changes.

If you follow good committing practices (like we all should do), you can simply use the command

`git add <name of the file to add to commit>`

to stage the changes in a given file for a commit.

Once you staged all the changes you want in a given commit, you can use the command

`git commit`

This will open a text editor in terminal for you to enter your commit message. Please follow the guideline on [how to commit](https://github.com/PinguDevTeam/ICF-MI-framework/formats/commits.md) while writing your commit messages. Once 

If you don't want to have a second line in your commit message, you can use

`git commit -m "<commit message>"`

to commit without the text editor interface.

In VSCode, this is much easier and is achieved through the `source control` menu on the left ribbon. Using that interface you can easily create, stage and commit your changes.

If you accidentally commited and want to undo your last commit, you can use the following command

`git reset --soft HEAD~1`

This will revert your last commit on the repository and get you to the point in time where you staged your changes. 

*Don't* commit a change where you modify 42 lines of code affecting the functionality of numerous functions. Scope out your work and commit them in bite-sized chunks. You can select which changes you want to stage from a file with numerous lines of change.


Once you have committed your changes with an appropriate [commit message](https://github.com/PinguDevTeam/ICF-MI-framework/formats/commits.md), push your changes when you are ready using the command

`git push`

This action will not push the code to the main branch, it will just update the code on your active branch. The commits you have made will be visible in the related PR.

Once you have committed your changes with an appropriate [commit message](https://github.com/PinguDevTeam/ICF-MI-framework/blob/91ffb3dd783ecc1843eb7a1314276d4ef1636d69/formats/commits.md), push your changes when you are ready. This action will not push the code to the main branch, it will just update the code on your active branch. The commits you have made will be visible in the related PR.