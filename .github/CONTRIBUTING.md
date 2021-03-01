# How to contribute

We welcome contributions from external contributors, and this document
describes how to merge code changes into this despasito. We welcome contributions from everyone in the form of bug fixes, new features, documentation and suggestions. 

## Release Procedure

Note that it is our goal that the master branch reflects current [pypi](https://pypi.org/project/despasito/) package. Thus all Pull Requests (PR) should be directed to the `update` branch. Periodically, a stable version of the `update` branch will be merged with the master branch as a new release.

## Getting Started

* Make sure you have a [GitHub account](https://github.com/signup/free).
* [Fork](https://help.github.com/articles/fork-a-repo/) this repository on GitHub.
* On your local machine,
  [clone](https://help.github.com/articles/cloning-a-repository/) your fork of
  the repository.

## Making Changes

* Add some really awesome code to your local fork.  It's usually a [good
  idea](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/)
  to make changes on a
  [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
  with the branch name relating to the feature you are going to add.
* When you are ready for others to examine and comment on your new feature,
  navigate to your fork of despasito on GitHub and open a [pull
  request](https://help.github.com/articles/using-pull-requests/) (PR). Note that
  after you launch a PR from one of your fork's branches, all
  subsequent commits to that branch will be added to the open pull request
  automatically.  Each commit added to the PR will be validated for
  mergability, compilation and test suite compliance; the results of these tests
  will be visible on the PR page.
* If you're providing a new feature, you must add test cases and documentation.
* When the code is ready to go, make sure you run the test suite using pytest.
* When you're ready to be considered for merging, check the "Ready to go"
  box on the PR page to let the despasito devs know that the changes are complete.
  The code will not be merged until this box is checked, the continuous
  integration returns check marks,
  and multiple core developers give "Approved" reviews.

## Change Stipulations

Updates will *only* be considered if:

 * The change is provided in a pull request via a branch with a meaningful name to the update branch.
   (Use python-compatible names.)
 * Hold to [PEP8](https://pep8.org/) Python style.
 * Uphold the current documentation standards both in the form of docstrings with the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) and proper inclusion in the docs directory.
    Note that a bash script, run.sh, has been included in the docs directory to rapidly generate the documentation.
 * An example is provided in a directory that is meaningfully named.
 * A test is included in an relevant existing or new test_\*.py file that doesn't involve an unwarranted increase in testing time.
 * Running pre-pull-request testing below exits with no errors and minimal code quality issues.
 
 I reserve the right to request you to modify the update as I see fit for any reason prior to inclusion. Modifications might be made for example:

* To provide consistent style or naming.
* To alleviate obvious code issues.
* Adjust to match other changes in play.
* etc.

## Suggested Tools

 * [yapf](https://pypi.org/project/yapf/) Autoformatter for PEP8 guidelines

# Additional Resources

* [General GitHub documentation](https://help.github.com/)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
* [Thinkful PR example](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)
* Package Architecture in documentation
