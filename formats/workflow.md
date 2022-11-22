## General Workflow

Each time you start to work on a branch, please `fetch` all the changes from the reposit using `git fetch` and check if there are any changes affecting your current active branch using `git status`.

If there are changes affecting your active branch. Please get all the changes from the relevant parent branch using the command

```git merge <name of child branch> origin/<path of the parent branch>```

Please commit those merges appropriately according to the naming convention

Once you work on a small task towards addressing a issue, please commit your changes in small packages. *Don't* commit a change where you modify 42 lines of code affecting the functionality of numerous functions. Scope out your work and commit them in bite-sized chunks. You can select which changes you want to stage from a file with numerous lines of change.

At any point in time if you require to see the HEAD state (pointer to your local) or the last commit that was made on the branch you can use ```git log```.

Once you have committed your changes with an appropriate [commit message](https://github.com/PinguDevTeam/ICF-MI-framework/blob/91ffb3dd783ecc1843eb7a1314276d4ef1636d69/formats/commits.md), push your changes when you are ready. This action will not push the code to the main branch, it will just update the code on your active branch. The commits you have made will be visible in the related PR.