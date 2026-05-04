#!/usr/bin/env python3
"""
One-command setup for SmartPhone Pose Estimation — builds the iOS app
and installs it onto a connected iPhone.

Usage:
    python start.py                    # auto-detect iPhone
    python start.py --device <UDID>    # specify device
    python start.py --skip-pods        # skip pod install (faster reruns)

Requirements (macOS only):
    - Xcode (from App Store)
    - Node.js 20+  (nvm install 20 && nvm use 20)
    - CocoaPods    (sudo gem install cocoapods)
    - iPhone connected via USB, unlocked, trusted
    - Free Apple ID configured in Xcode (Settings → Accounts)
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ─────────────────────────── Console helpers ───────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[1;36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def step(msg: str) -> None:
    print(f"\n{CYAN}▶ {msg}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def die(msg: str) -> None:
    print(f"\n{RED}✗ {msg}{RESET}", file=sys.stderr)
    sys.exit(1)


def run(cmd: str | list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Stream a subprocess. Dies on nonzero exit."""
    shell = isinstance(cmd, str)
    pretty = cmd if shell else " ".join(cmd)
    print(f"  $ {pretty}")
    result = subprocess.run(cmd, shell=shell, cwd=cwd, env=env)
    if result.returncode != 0:
        die(f"command failed (exit {result.returncode}): {pretty}")


# ─────────────────────────── Prerequisite checks ───────────────────────────
def check_mac() -> None:
    if platform.system() != "Darwin":
        die("This script only runs on macOS (Xcode is required for iOS builds).")


def check_xcode() -> None:
    if not shutil.which("xcodebuild"):
        die("Xcode not found. Install from the App Store, then run: "
            "sudo xcode-select --install")
    ok("Xcode installed")


def check_node() -> None:
    node = shutil.which("node")
    if not node:
        die("Node.js not found. Install from https://nodejs.org or: brew install node")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    major = int(version.lstrip("v").split(".")[0])
    if major < 20:
        die(f"Node {version} is too old. Need Node 20+. Try:\n"
            "  source ~/.nvm/nvm.sh && nvm install 20 && nvm use 20")
    ok(f"Node {version}")


def check_pod() -> None:
    if not shutil.which("pod"):
        die("CocoaPods not found. Install: sudo gem install cocoapods")
    ok("CocoaPods installed")


# ─────────────────────────── Dependency install ───────────────────────────
def npm_install() -> None:
    step("Installing JS dependencies (npm install)")
    if (HERE / "node_modules").exists():
        ok("node_modules exists — skipping (rm -rf node_modules to force reinstall)")
        return
    run(["npm", "install"], cwd=HERE)


def pod_install() -> None:
    step("Installing CocoaPods (with UTF-8 locale fix)")
    # Ruby's UnicodeNormalize rejects ASCII-8BIT; LANG/LC_ALL forces UTF-8
    env = os.environ.copy()
    env["LANG"] = "en_US.UTF-8"
    env["LC_ALL"] = "en_US.UTF-8"
    run(["pod", "install", "--repo-update"], cwd=HERE / "ios", env=env)


# ─────────────────────────── iPhone detection ───────────────────────────
def list_iphones() -> list[tuple[str, str]]:
    """Returns [(udid, description), ...] of connected, unlocked iPhones."""
    r = subprocess.run(
        ["xcrun", "devicectl", "list", "devices"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        die(f"xcrun devicectl failed: {r.stderr.strip()}")

    devices: list[tuple[str, str]] = []
    for line in r.stdout.splitlines():
        if "connected" not in line or "iPhone" not in line:
            continue
        # UDID is 36 chars with 4 dashes
        for token in line.split():
            if len(token) == 36 and token.count("-") == 4:
                devices.append((token, line.strip()))
                break
    return devices


def pick_iphone(requested_udid: str | None) -> str:
    step("Locating iPhone")
    devices = list_iphones()
    if not devices:
        die("No iPhone detected. Plug in iPhone via USB, unlock it, "
            "and 'Trust' this Mac when prompted.")

    if requested_udid:
        matches = [d for d in devices if d[0].lower() == requested_udid.lower()]
        if not matches:
            die(f"UDID {requested_udid} not found among connected devices.")
        ok(f"Using requested device: {matches[0][1]}")
        return matches[0][0]

    if len(devices) == 1:
        ok(f"Found: {devices[0][1]}")
        return devices[0][0]

    print("  Multiple iPhones connected — pick one:")
    for i, (udid, line) in enumerate(devices):
        print(f"    [{i}] {line}")
    while True:
        choice = input("  Choice: ").strip()
        try:
            return devices[int(choice)][0]
        except (ValueError, IndexError):
            print("  Invalid choice, try again.")


# ─────────────────────────── Build + install ───────────────────────────
def find_built_app() -> Path:
    """Locate the .app bundle produced by the last Debug-iphoneos build."""
    settings = subprocess.check_output(
        ["xcodebuild",
         "-workspace", "ios/FirstApp.xcworkspace",
         "-scheme", "FirstApp",
         "-configuration", "Debug",
         "-sdk", "iphoneos",
         "-showBuildSettings"],
        cwd=HERE, text=True,
    )
    build_dir = None
    for raw in settings.splitlines():
        line = raw.strip()
        if line.startswith("BUILT_PRODUCTS_DIR"):
            build_dir = line.split("=", 1)[1].strip()
            break
    if not build_dir:
        die("Could not locate BUILT_PRODUCTS_DIR from xcodebuild settings.")
    app_path = Path(build_dir) / "FirstApp.app"
    if not app_path.exists():
        die(f"Built app not found at {app_path}. Did the build fail?")
    return app_path


def build_and_install(udid: str) -> None:
    step(f"Building iOS app for device {udid}")
    run(
        [
            "xcodebuild",
            "-workspace", "ios/FirstApp.xcworkspace",
            "-scheme", "FirstApp",
            "-configuration", "Debug",
            "-destination", f"id={udid}",
            "-allowProvisioningUpdates",
            "build",
        ],
        cwd=HERE,
    )
    ok("Build succeeded")

    step("Installing on iPhone")
    app_path = find_built_app()
    run(
        ["xcrun", "devicectl", "device", "install", "app",
         "--device", udid, str(app_path)]
    )
    ok("App installed on iPhone")


# ─────────────────────────── Main ───────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="UDID of target iPhone (default: auto-detect)")
    parser.add_argument("--skip-pods", action="store_true",
                        help="Skip pod install (use after first run)")
    parser.add_argument("--skip-npm", action="store_true",
                        help="Skip npm install")
    args = parser.parse_args()

    print(f"{BOLD}SmartPhone Pose Estimation — setup{RESET}\n")

    step("Checking prerequisites")
    check_mac()
    check_xcode()
    check_node()
    check_pod()

    if not args.skip_npm:
        npm_install()
    if not args.skip_pods:
        pod_install()

    udid = pick_iphone(args.device)
    build_and_install(udid)

    print(f"\n{GREEN}{BOLD}Done!{RESET}\n")
    print("On your iPhone:")
    print("  1. Open the 'SmartPhone Pose Estimation' app (purple gradient icon)")
    print("  2. First time only: Settings → General → VPN & Device Management →")
    print("     trust your developer profile")
    print("  3. The Server URL should already show the HF Space URL")
    print("  4. Record a short clip and tap 'Upload & Run MeTRAbs'")
    print("\nBackend (already running):")
    print("  https://engdesfau26-smartphonepose-metrabs-server.hf.space")


if __name__ == "__main__":
    main()
