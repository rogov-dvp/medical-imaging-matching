# Testing Plan

Updated: October 21st, 2021 by [Emily Medema](https://github.com/emedema)

Reason for Update: Adding developments from Requirements Milestone and Group Evaluation

## Testing Frameworks

### Unit Tests
We will be implementing unit testing throughout our project. Each method in the model(s) as well as within our preprocessing script will have a unit test with which it is associated. All of our unit tests will utilize the built-in python testing library `unittest`. This testing will ensure that all of our methods are functioning and returning the values we expect.

### Integration Testing
In addition to unit testing, we will also be performing integration testing on our codebase. This is an essential element to our testing as there are multiple scripts and interesting data requirements within this project. We need to ensure that the model script and the preprocessing script are working together properly; that the data is accessed safely and securely; and that the data is not leaving BC Cancer’s network.

[Integration Testing Resource](https://mq-software-carpentry.github.io/python-testing/09-integration/)

### Regression Testing
Likewise, we will also be performing regression testing. This will ensure we are not breaking our code whenever we add a new piece of code. To do this, we will utilize the `pytest-regtest` python plugin to check that the format of the output remains the same as before (ie. no errors).

[Regression Testing Resource](https://pypi.org/project/pytest-regtest/)

### Functional Testing
Lastly, we will be performing functional testing. It is essential to our project that we select the correct images in accordance with the mismatches we input and that we return a similarity index that makes sense. Functional testing will help us ensure that this is true.

## CI/CD

We will also be implementing Continuous Integration/Continuous Development procedures within our project to ensure our code does not break when we make changes. We will run premade test cases on each model we develop. These test cases include a set of mammograms in which we already know the results (mammograms match or not). If the success rate is too low, we will know we must improve the model. We will also configure static analysis, specifically the `pre-commit` framework so that our code is consistently formatted and linted. This will run via git-hooks and will catch things like whitespace and common issues.

[`Pre-Commit` Framework Documentation](https://pre-commit.com/#cli)

[CI/CD Tutorial](https://towardsdatascience.com/ci-cd-by-example-in-python-46f1533cb09d)

### CI Workflow
On Github we will be following the workflow detailed below to ensure CI:
1. We will be creating issues for each feature, separated into the task, chore, and exploration labels depending on the work needed for each issue.
    - Tasks: A task is a step in the process of completing a feature. A feature is completed when underlying tasks are completed. Usually, tasks will include many aspects such as coding, testing, and documentation.
    - Chores: A chore is an item that provides value to the team rather than the client. Examples include implementing continuous integration, linting the project, or refactoring a codebase.
    - Explorations: Exploration is required when breaking a feature into tasks is not possible, either because you are unclear what needs to be done or unclear how it should be accomplished. This is the label used when additional research needs to be done for us to determine how to move forward.
2. One person on the team, unless otherwise specified, will be assigned at least two issues per sprint to work on.
3. Once the assignee determines the work to be done, they will pull and merge any changes that occurred on the main branch into their feature and they will run the test cases that were created. Once those tests pass, they will create a pull request.
4. The pull request will be reviewed by at least two other team members. All comments and concerns must be addressed and the assignee must be given the explicit go-ahead to merge.
5. After the merging of a PR, the main codebase will be subjected to the premade test cases and everyone will be notified that they need to pull to keep their local repository up to date. **Note**: When the PR is merged the issue is automatically closed.
