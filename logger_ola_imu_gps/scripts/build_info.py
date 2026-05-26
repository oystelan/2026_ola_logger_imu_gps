"""PlatformIO pre-build script that injects build-time metadata as -D flags.

Replaces the Unix-only `!echo "-DKEY="$(cmd)` lines in platformio.ini so the
build works on Windows, Linux and macOS.
"""

import getpass
import os
import socket
import subprocess

Import("env")  # noqa: F821 (provided by SCons)


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


project_dir = env["PROJECT_DIR"]

git_branch = _run(["git", "-C", project_dir, "rev-parse", "--abbrev-ref", "HEAD"])
commit_id = _run(["git", "-C", project_dir, "rev-parse", "HEAD"])
repo_toplevel = _run(["git", "-C", project_dir, "rev-parse", "--show-toplevel"])
repo_basename = os.path.basename(repo_toplevel) if repo_toplevel != "unknown" else "unknown"

try:
    host_name = socket.gethostname() or "unknown"
except Exception:
    host_name = "unknown"

try:
    user_name = getpass.getuser() or "unknown"
except Exception:
    user_name = "unknown"

project_name = os.path.basename(os.path.dirname(project_dir))

env.Append(
    BUILD_FLAGS=[
        "-DREPO_GIT_BRANCH=" + git_branch,
        "-DREPO_COMMIT_ID=" + commit_id,
        "-DCOMPILING_HOST_NAME=" + host_name,
        "-DCOMPILING_USER_NAME=" + user_name,
        "-DREPO_BASENAME=" + repo_basename,
        "-DPROJECT_NAME=" + project_name,
    ]
)
