# Contributing {#contributing}

Thanks for contributing 🙂‍↕️

## Coding style

Please [format your code](https://clang.llvm.org/docs/ClangFormat.html) with `clang-format`
**before committing**
(the project ships a `.clang-format` file).

Also, please roughly adapt to the coding style of the project.
Short overview (you may as well inspect the code):
```cpp
namespace terra::grid::shell {      // mimic directory structure - use this for a file in src/terra/grid/shell/ 

// snake_case for functions, argument, and variables

float some_function( float some_arg )
{
    float some_var = 1.0f;
    return some_arg + some_var;
}

// CamelCase for classes

class SomeClass
{
  public:
    void some_method() { /* ... */ } // snake_case for methods.

  private:
    int some_private_member_; // underscore at the end for (private) member variables
};

struct SomeDataStruct
{
    int some_struct_member; // if public and simply in a container - no underscore at the end
};

}
```

## Pull requests

> TL;DR: We are employing a simple fork workflow on the main branch.

Here's a short guide on how to get started.

**Some tips**:
- Don’t worry about mistakes. We can help fix them.
- Small pull requests are easier to review than large ones.
- You can draft a PR early to get feedback.

If unsure, follow the steps below.
For more details, you can, for instance, check out the 
[Atlassian guide](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) for more details.

#### 1. Fork the repository

At the top right of the GitHub page click:
```
Fork → Create fork
```
This makes your own copy of the project.

#### 2. Clone your fork

Open a terminal:
```
git clone https://github.com/<your-username>/TERRA-NG.git
cd TERRA-NG
```

#### 3. Set the original repo as “upstream” (one-time setup)

This lets you pull updates from the main project:
```
git remote add upstream https://github.com/mantleconvection/TERRA-NG.git
```
You only do this once.

#### 4. Always start from the latest main branch

Before creating new work:
```
git fetch upstream
git checkout main
git pull upstream main
```

#### 5. Create a new branch for your change

Never work directly on main.
```
git checkout -b feature/my-change
```

Pick any name that describes your change
(e.g., bugfix/mpi-deadlock, feature/some-forcing-term, docs/improving-fem-documentation).

#### 6. Make your changes

Make the edits you want.

\note Please adhere to the coding style of the project and format with `clang-format` (the project ships a `.clang-format` file).

Then save your work:
```
git add .
git commit -m "Describe your change."
```
Keep commit messages simple and clear and so that others can directly see what you have changed.

#### 7. Update your branch before pushing (important!)

Make sure your branch has the latest updates from the project:
```
git fetch upstream
git pull upstream main
```

If it asks about conflicts, fix them if you can.
If you’re unsure, ask a maintainer. We’re happy to help.

#### 8. Push your branch to your fork
```
git push origin feature/my-change
```

#### 9. Open a Pull Request (PR)

Go to your fork on GitHub.
GitHub will show a “Compare & pull request” button.

Make sure the PR target is:
```
base: main   ←   compare: feature/my-change
```

In your PR description:
- explain what you changed
- mention related issues (if any)

That’s it!

A maintainer will review the PR and merge it into main.
