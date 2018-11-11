# Contributing to IPNeuronaleNetze

We'd love for you to contribute to our source code and to make our project even better than it is
today! Here are the guidelines we'd like you to follow:

* Issues and Bugs
* Improving Documentation
* Issue Submission Guidelines
* Pull Request Submission Guidelines
* Coding Rules
* Git Commit Guidelines
* Writing Documentation

## Bugs

### Found an Issue or Bug?

If you find a bug in the source code, you can help us by submitting an issue to our
[GitHub Repository](https://github.com/Paeti/IPNeuronaleNetze). Even better, you can submit a Pull Request with a fix.

**Please see the Submission Guidelines below.**

### Want a Doc Fix?

Should you have a suggestion for the documentation, you can open an issue and outline the problem
or improvement you have - however, creating the doc fix yourself is much better!

If you want to help improve the docs, it's a good idea to let others know what you're working on to
minimize duplication of effort. Create a new issue (or comment on a related existing one) to let
others know what you're working on.

You should make sure that your commit message follows the Commit Message Guidelines.

## Issue Submission Guidelines

Ensure the bug was not already reported by searching on GitHub under issues.
If you're unable to find an open issue addressing the bug, open a new issue.

In general, providing the following information will increase the chances of your issue being dealt
with quickly:

* **Overview of the Issue** - describe what has to be done
* **Reproduce the Error** - if it's an error,provide a example where it fails
* **Related Issues** - has a similar issue been reported before?
* **Suggest a Fix** - if you can't fix the bug yourself, perhaps you can point to what might be
  causing the problem (line of code or commit)
* **Tag the issue** - tag it with an existing tag or create a matching one

**If you get help, help others. Good karma rulez!**

## Pull Request Submission Guidelines

Each pull request to the master refers to one issue and closes it.
Before you submit your pull request consider the following guidelines:

* Name pull request as following:
- <tag of the ticket>/<__really__ short description waht you'll do>_<nr of ticket it should close>
- e.g. enh/create_file/method_xy_7
* Make your changes in a new git branch:

    ```shell
    git checkout -b my-fix-branch master
    ```
* Name my-fix-branch following the same rules as pull requests.
* Create your patch commit, **including appropriate test cases**.
* Follow our Coding Rules.
* If the changes affect public APIs, change or add relevant documentation.
* Commit your changes using a descriptive commit message that follows our
  commit message conventions. Adherence to the
  commit message conventions is required.

    ```shell
    git commit -a
    ```
  Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

* Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

* First the commit has to pass the continous integration test. We choose drone as our ci tool.

* If you find that the drone integration has failed, look into the logs on drone to find out
if your changes caused test failures etc.

* If we suggest changes, then:

  * Make the required updates
  * Commit your changes to your branch (e.g. `my-fix-branch`).
  * Push the changes to your GitHub repository (this will update your Pull Request).

    You can also amend the initial commits and force push them to the branch.

    ```shell
    git rebase master -i
    git push origin my-fix-branch -f
    ```

    This is generally easier to follow, but seperate commits are useful if the Pull Request contains
    iterations that might be interesting to see side-by-side.

That's it! Thank you for your contribution!

#### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes
from the main (upstream) repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```

* Check out the master branch:

    ```shell
    git checkout master -f
    ```

* Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```


## Coding Rules

We follow the PEP-8 styleguide. There are some tools out there that formats your code PEP-8 conform.
E.g. for [visual studio code](https://marketplace.visualstudio.com/items?itemName=himanoa.Python-autopep8)

If you work on the tensorflow part of the project also follow these [guidelines](https://danijar.com/structuring-your-tensorflow-models/).

## Git Commit Guidelines

We have very precise rules over how our git commit messages can be formatted.  This leads to **more
readable messages** that are easy to follow when looking through the **project history**.

### Commit Message Format
Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special
format that includes a **type**, a **scope** and a **subject**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory and the **scope** of the header is optional.

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on GitHub as well as in various git tools.

### Revert
If the commit reverts a previous commit, it should begin with `revert: `, followed by the header
of the reverted commit.
In the body it should say: `This reverts commit <hash>.`, where the hash is the SHA of the commit
being reverted.
A commit with this format is automatically created by the `git revert` command.

### Type
Must be one of the following:

* **enh**: A new enhancement
* **fix**: A bug fix
* **docs**: Documentation only changes
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing
  semi-colons, etc)
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing or correcting existing tests
* **chore**: Changes to the build process or auxiliary tools and libraries such as documentation
  generation

### Scope
The scope could be anything specifying place of the commit change. For example `$models`,
`$trainers`, `$web`, `$global`, `Readme`, `trainers`, `.gitignore`, etc...

You can use `*` when the change affects more than a single scope.

### Subject
The subject contains succinct description of the change:

### Body
The body should include the motivation for the change and contrast this with previous behavior.

### Footer
The footer is the place to reference GitHub issues that this commit closes.
Closing issues using keywords

You can include keywords in your pull request descriptions, as well as commit messages, to automatically close issues in GitHub.

When a pull request or commit references a keyword and issue number, it creates an association between the pull request and the issue.
When the pull request is merged into your repository's default branch, the corresponding issue is automatically closed.

The following keywords, followed by an issue number, will close the issue:

    close
    closes
    closed
    fix
    fixes
    fixed
    resolve
    resolves
    resolved

For example, to close an issue numbered 123, you could use the phrase "Closes #123" or "Closes: #123" in your pull request description
or commit message. Once the branch is merged into the default branch, the issue will close.

### Example
enh(Readme): Create Readme

To give the people a short overview over the project it should have a Readme.
Should contain:
- Motivation
- License
- ....

closes #1


This commit is about an enhancement and has the scope Readme. It's not always easy to say which scope the commit has.
Has the Readme impact just on itsself or global? So choose what you think is best and if the others think different they'll
correct you.
The last line says that this commit closes the ticket #1.

## Writing Documentation

For each issue with the __research__ tag a single markdown file is created in the /docs folder.
All people assigned to that issue report their results in this file.
The bigger modules of the project also get a markdown file. A documentation including what the method is doing and how to use it
has to be written for each one. This must be done dirrectly after finishing the method.
