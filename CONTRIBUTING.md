# Contributing

Contributions (Pull Requests) are very much welcome! Here are a few guidelines on how to do it.

## Getting started

First, clone the repo and install the package in developing mode

```bash
git clone git@github.com:arnauqb/blackbirds.git
cd blackbirds
pip install -e .
```

## Contributing to the code

If you plan on contributing to the code, the recommended way is to follow a test-driven approach, preferably writing the tests before the actual code (;-)).
We will reject PRs that are not properly tested, you should aim to mantain a similar level of coverage to the rest of the package. We use PyTest as a test-suite, you can run the current unit tests by doing

```bash
pip install pytest pytest-cov
pytest test
```

Your code should be formatted in small functions, each individually tested in the test folder.

## Code conventions

We use the [black](https://github.com/psf/black) Python formatter. Make sure that all code you push is formatted properly.
Function and variable names should be self explanatory. Single letter variables should be avoided unless their meaning is clear by context.
Functions' docstrings should follow the rest of the code's format. This makes it easier to build documentation automatically.

## Opening a Pull Request

Once you are happy with the local changes to the code, which should have been done in a separate git branch specific to the issue, you can push your branch to the repository and open a pull request through the web interface. Make sure to please add details on the PR description about the changes you incorporated. The PR will automatically trigger a run of the unit tests, the PR wil not be approved until all tests pass.

## Contributing to the documentation

We use `mkdocs` as our documentation tool. You can install the requirements through

```bash
pip install -r docs/requirements.txt
```

To build the docs, simply do

```bash
mkdocs build
```

and you can deploy them locally with

```bash
mkdocs serve
```

This will open a local server at `localhost:8000` that you can use to preview your changes. The docs are automatically build and deployed with each PR using the respective github action, so there is no need to push or upload any build files.

