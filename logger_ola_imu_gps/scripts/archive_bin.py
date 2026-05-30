"""PlatformIO post-build script that archives the freshly-built firmware.bin
under a date- and git-SHA-tagged name in the project root, so each successful
build leaves a self-describing artefact next to platformio.ini.

Output name: `firmware_<YYYY-MM-DD>_<git_short_sha>.bin`.
If the build is from a working tree with uncommitted changes the suffix
`_dirty` is appended; if not inside a git repo the short-sha part becomes
`nogit`. Existing archives are silently overwritten when SHA + date match.
"""

import datetime
import os
import shutil
import subprocess

Import("env")  # noqa: F821 (provided by SCons)


def _run(cmd, **kwargs):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, **kwargs)
        return out.decode("utf-8", errors="replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _archive(source, target, env):
    project_dir = env["PROJECT_DIR"]
    src = os.path.join(env.subst("$BUILD_DIR"), "firmware.bin")
    if not os.path.isfile(src):
        print(f"[archive_bin] firmware.bin not found at {src}; skipping")
        return

    sha = _run(["git", "-C", project_dir, "rev-parse", "--short", "HEAD"]) or "nogit"
    dirty = _run(["git", "-C", project_dir, "status", "--porcelain"])
    if sha != "nogit" and dirty:
        sha = f"{sha}_dirty"

    date = datetime.date.today().isoformat()
    dst_name = f"firmware_{date}_{sha}.bin"
    dst = os.path.join(project_dir, dst_name)

    shutil.copy2(src, dst)
    print(f"[archive_bin] {dst_name}")


env.AddPostAction("$BUILD_DIR/firmware.bin", _archive)  # noqa: F821
