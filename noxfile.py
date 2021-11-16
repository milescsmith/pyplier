import tempfile
from typing import Any
from nox_poetry import session

import nox
from nox.sessions import Session

package = "pyplier"
nox.options.sessions = ["black", "isort", "tests"]  # "mypy", "pytype", "lint",
locations = (
    "src",
    "tests",
)  # "noxfile.py", #"docs/conf.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """Install packages constrained by Poetry's lock file.
    This function is a wrapper for nox.sessions.Session.install. It
    invokes pip to install packages inside of the session's virtualenv.
    Additionally, pip is passed a constraints file generated from
    Poetry's lock file, to ensure that the packages are pinned to the
    versions specified in poetry.lock. This allows you to manage the
    packages as Poetry development dependencies.
    Arguments:
        session: The Session object.
        args: Command-line arguments for pip.
        kwargs: Additional keyword arguments for Session.install.
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@session(python=["3.9"])
def tests(session: Session) -> None:
    args = session.posargs or locations

    session.install("pytest", ".")
    session.install("hypothesis")
    session.run("pytest", *args)


@session(python=["3.9"])
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black[jupyter]")
    session.run("black", *args)


@session(python=["3.9"])
def isort(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("isort")
    session.run("isort", *args)


@session(python=["3.9"])
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)
