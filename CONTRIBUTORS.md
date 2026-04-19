# Contributor procedure
To contribute you must create a Pull Request (PR) in this repository. Using your computer terminal follow the commands below to create a new branch, stage and commit your changes and finally open a PR.

- Create and checkout new working branch. Name of your branch cannot have spaces, use hyphens. 
```
git checkout -b <branch-name-here>
```
- Make edits
- Stage all your changes
```
git add .
```
- Commit
```
git commit -m "<commit-name-usually-an-action>"
```
- Push branch to GitHub
```
git push origin <branch-name-here>
```
- On GitHub.com click on "Create a new Pull Request"
- For `base` choose `main`. Use your newly created branch in `compare`.

## Creating/Editing a notebook
If your PR includes a Jupyter notebook (`ipynb` file): before opening a PR please export your new or edited notebook to a Python file so all contributors can review the changes from Github directly 

Export Python file with
```
jupyter nbconvert --to script --ClearOutputPreprocessor.enabled=True your_notebook.ipynb
```

Alternatively you can explort your notebook as py using the Jupyter or VS code UI. 

- Jupyter: File>Save and Export Notebook as>Executable Script
- VS code: In the same menu where you find `+ Code` and `Run all`, click on the three dots `...` located to the right. Click `export` and then choose `Python`. 


