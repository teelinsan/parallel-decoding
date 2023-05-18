import logging
import os
from pathlib import Path
from typing import Optional

import dotenv
import git

pylogger = logging.getLogger(__name__)


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """Safely read an environment variable.
    Raises errors if it is not defined or it is empty.
    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            message = f"{env_name} not defined and no default value is present!"
            pylogger.error(message)
            raise KeyError(message)
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            message = (
                f"{env_name} has yet to be configured and no default value is present!"
            )
            pylogger.error(message)
            raise ValueError(message)
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environment variables
load_envs()


if "PROJECT_ROOT" not in os.environ:
    try:
        PROJECT_ROOT = Path(
            git.Repo(Path.cwd(), search_parent_directories=True).working_dir
        )
    except git.exc.InvalidGitRepositoryError:
        PROJECT_ROOT = Path.cwd()

    pylogger.debug(f"Inferred project root: {PROJECT_ROOT}")
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
else:
    PROJECT_ROOT: Path = Path(os.environ["PROJECT_ROOT"])

__all__ = ["__version__", "PROJECT_ROOT"]
