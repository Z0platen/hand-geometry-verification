"""Desktop launcher for combined hand verification."""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent


def validate_profile_id(value, field="User ID"):
    if isinstance(value, str) and not value.strip():
        if field == "Actual person ID":
            raise ValueError("Enter the Actual person ID. Use the selected profile's ID if you are testing your own hand.")
        raise ValueError(f"Enter the {field}.")
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value) is None:
        raise ValueError(f"{field} must contain 1–32 letters, numbers, underscores, or hyphens.")
    return value


def camera_index(value):
    text = str(value).strip()
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("Camera must be a non-negative whole number, such as 0 or 1.")
    return int(text)


def condition_label(value):
    if not isinstance(value, str):
        raise ValueError("Enter a condition label, such as daylight or low_light.")
    value = value.strip()
    if not value or len(value) > 80 or not value.isprintable():
        raise ValueError("Condition must be 1–80 characters on one line.")
    return value


@dataclass(frozen=True)
class Profile:
    user_id: str
    model_path: Path | None
    sample_count: int | None
    session_count: int
    palm_model_path: Path | None = None


def list_profiles(captures_root):
    """Read completed-session metadata only; never open biometric images."""
    captures_root = Path(captures_root).resolve()
    if not captures_root.is_dir():
        return []
    profiles = []
    for user_directory in sorted(captures_root.iterdir(), key=lambda path: path.name.casefold()):
        try:
            user_id = validate_profile_id(user_directory.name)
        except ValueError:
            continue
        if not user_directory.is_dir() or not user_directory.resolve().is_relative_to(captures_root):
            continue
        models = [
            path for path in user_directory.glob("*/model.json")
            if path.is_file() and path.resolve().is_relative_to(captures_root)
        ]
        palm_models = [path for path in user_directory.glob("*/palm_model.json")
                       if path.is_file() and path.resolve().is_relative_to(captures_root)]
        if not models and not palm_models:
            continue
        latest_geometry = max(models, key=lambda path: path.parent.name) if models else None
        latest_palm = max(palm_models, key=lambda path: path.parent.name) if palm_models else None
        latest = latest_palm or latest_geometry
        sample_count = None
        try:
            model = json.loads(latest.read_text(encoding="utf-8"))
            count = model.get("enrollment_counts", {}).get(user_id) if isinstance(model, dict) else None
            if type(count) is int and count > 0:
                sample_count = count
        except (OSError, UnicodeError, ValueError, AttributeError):
            pass
        session_count = len({path.parent for path in models + palm_models})
        profiles.append(Profile(user_id, latest_geometry, sample_count, session_count, latest_palm))
    return profiles


def build_command(
    action, user_id, *, camera="0", image_path=None, actual_id="",
    condition="standard", root=ROOT, executable=None, distance_cm="", scale="1.0",
):
    """Construct separate arguments; user-entered values never become shell code."""
    if action not in ("enroll", "live", "photo", "experiment"):
        raise ValueError(f"Unknown action: {action!r}.")
    root = Path(root).resolve()
    executable = str(Path(executable or sys.executable).resolve())
    user_id = validate_profile_id(user_id, "New user ID" if action == "enroll" else "Selected profile ID")
    command = [executable, str(root / "palm_demo.py"), f"--user={user_id}"]
    if action == "photo":
        if image_path is None or not Path(image_path).is_file():
            raise ValueError("Select an existing image file.")
        return command + ["--image", str(Path(image_path).resolve()), "--show"]

    command.extend(["--camera", str(camera_index(camera))])
    if action == "enroll":
        return command + ["--reenroll"]
    if action == "live":
        return command

    actual_id = validate_profile_id(actual_id, "Actual person ID")
    try:
        factor = float(scale)
    except (ValueError, TypeError) as error:
        raise ValueError("Input scale must be between 0.1 and 1.0.") from error
    if isinstance(scale, bool) or not math.isfinite(factor) or not 0.1 <= factor <= 1:
        raise ValueError("Input scale must be between 0.1 and 1.0.")
    command.extend([f"--actual={actual_id}", f"--condition={condition_label(condition)}", f"--scale={factor:g}"])
    if distance_cm is not None and str(distance_cm).strip():
        try:
            distance = float(distance_cm)
        except (ValueError, TypeError) as error:
            raise ValueError("Measured distance must be a positive number in centimeters.") from error
        if isinstance(distance_cm, bool) or not math.isfinite(distance) or distance <= 0:
            raise ValueError("Measured distance must be a positive number in centimeters.")
        command.append(f"--distance-cm={distance:g}")
    return command


@dataclass(frozen=True)
class RunResult:
    label: str
    returncode: int
    log_path: Path
    detail: str


def process_error_detail(text):
    lines = text.splitlines()
    prefixes = (
        "Verification error:", "Enrollment error:", "Experiment error:", "Palm study error:", "Dataset demo error:",
    )
    for line in reversed(lines):
        if line.startswith(prefixes):
            return line
    return "\n".join(lines[-12:])


class ChildRunner:
    """Start at most one child, then let the Tk event loop poll its result."""

    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.process = None
        self.log_path = None
        self.label = None

    def start(self, command, label):
        if self.process is not None:
            raise ValueError("An action is already open. Close its window with Q or Esc first.")
        log_directory = self.root / "outputs" / "launcher"
        log_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_directory / f"{stamp}-{secrets.token_hex(4)}.log"
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        with log_path.open("xb") as log:
            log.write(f"{label}\n\n".encode("utf-8"))
            log.flush()
            try:
                process = subprocess.Popen(
                    command, cwd=self.root, shell=False,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except OSError as error:
                log.write(f"Could not start process: {error}\n".encode("utf-8"))
                raise
        self.process = process
        self.log_path = log_path
        self.label = label
        return log_path

    def poll(self):
        if self.process is None:
            return None
        returncode = self.process.poll()
        if returncode is None:
            return None
        detail = ""
        try:
            with self.log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 6000))
                detail = process_error_detail(log.read().decode("utf-8", errors="replace"))
        except OSError as error:
            detail = f"Could not read the process log: {error}"
        result = RunResult(self.label, returncode, self.log_path, detail)
        self.process = None
        self.log_path = None
        self.label = None
        return result


class Launcher:
    def __init__(self, window):
        self.window = window
        self.runner = ChildRunner()
        self.profiles = {}
        self.busy_controls = []
        self.profile_pickers = []
        self.selected = tk.StringVar()
        self.new_user = tk.StringVar()
        self.actual = tk.StringVar()
        self.condition = tk.StringVar(value="standard")
        self.distance = tk.StringVar()
        self.scale = tk.StringVar(value="1.0")
        self.camera = tk.StringVar(value="0")
        self.summary = tk.StringVar()
        self.status = tk.StringVar(value="Ready.")
        self.pending_enrollment = None
        window.title("Hand Verification")
        window.minsize(540, 460)
        window.geometry("680x490")
        window.protocol("WM_DELETE_WINDOW", self.close)

        body = ttk.Frame(window, padding=24)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Hand Verification", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(body, text="Combines hand shape and palm texture in one check.").pack(anchor="w", pady=(4, 16))
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill="both", expand=True)
        self.verify_tab = self.add_tab("Verify")
        self.enroll_tab = self.add_tab("Enroll")
        experiment = self.add_tab("Experiments")

        self.add_profile_picker(self.verify_tab)
        ttk.Label(self.verify_tab, textvariable=self.summary, wraplength=450).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0),
        )
        actions = ttk.Frame(self.verify_tab)
        actions.grid(row=2, column=0, columnspan=2, sticky="w", pady=(16, 0))
        self.add_button(actions, "Start camera", self.live).pack(side="left", padx=(0, 10))
        self.add_button(actions, "Choose photo", self.photo).pack(side="left")

        self.add_entry(self.enroll_tab, "New user ID", self.new_user, row=0)
        ttk.Label(
            self.enroll_tab, text="Register your own hand with 10 reference photos. The recognition model is already trained on 11K Hands.",
            wraplength=450,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.add_button(self.enroll_tab, "Enroll hand", self.enroll).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(16, 0),
        )

        self.add_profile_picker(experiment)
        self.add_entry(experiment, "Actual person ID", self.actual, row=1)
        self.add_entry(experiment, "Condition", self.condition, row=2)
        self.add_entry(experiment, "Measured distance (cm, optional)", self.distance, row=3)
        self.add_entry(experiment, "Digital input scale", self.scale, row=4)
        self.add_entry(experiment, "Camera index", self.camera, row=5)
        self.add_button(experiment, "Run experiment", self.experiment).grid(
            row=6, column=1, sticky="w", pady=(14, 0),
        )

        ttk.Label(body, textvariable=self.status, wraplength=450).pack(anchor="w", pady=(12, 0))
        self.refresh()
        window.after(250, self.poll)

    def add_tab(self, title):
        tab = ttk.Frame(self.tabs, padding=16)
        tab.columnconfigure(1, weight=1)
        self.tabs.add(tab, text=title)
        return tab

    def add_profile_picker(self, tab):
        ttk.Label(tab, text="Profile").grid(row=0, column=0, sticky="w", padx=(0, 12))
        picker = ttk.Combobox(tab, textvariable=self.selected, state="readonly")
        picker.grid(row=0, column=1, sticky="ew")
        picker.bind("<<ComboboxSelected>>", lambda _event: self.update_summary())
        self.profile_pickers.append(picker)

    def add_entry(self, tab, label, variable, *, row):
        spacing = (8, 0) if row else 0
        ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=spacing)
        entry = ttk.Entry(tab, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=spacing)
        self.busy_controls.append(entry)

    def add_button(self, parent, label, callback):
        button = ttk.Button(parent, text=label, command=callback)
        self.busy_controls.append(button)
        return button

    def refresh(self):
        try:
            self.profiles = {profile.user_id: profile for profile in list_profiles(ROOT / "captures")}
        except OSError as error:
            messagebox.showerror("Could not list profiles", str(error), parent=self.window)
            return
        for picker in self.profile_pickers:
            picker["values"] = tuple(self.profiles)
        if self.selected.get() not in self.profiles:
            self.selected.set("")
        self.update_summary()

    def update_summary(self):
        profile = self.profiles.get(self.selected.get())
        if profile is None:
            message = "Choose the profile you want to verify." if self.profiles else "Enroll a hand to create your first profile."
        elif profile.palm_model_path is None:
            message = "Update this profile under Enroll, or select Start camera to collect new samples."
        else:
            message = "Show your open right hand in the camera for live verification."
        self.summary.set(message)

    def selected_user(self, *, require_enrollment=False):
        profile = self.profiles.get(self.selected.get())
        if profile is None:
            raise ValueError("Choose a profile. If you have none, enroll a hand first.")
        if require_enrollment and profile.palm_model_path is None:
            raise ValueError("Update this profile under Enroll before using this action.")
        return profile.user_id

    def set_busy(self, busy):
        for widget in self.busy_controls:
            widget.configure(state="disabled" if busy else "normal")
        for picker in self.profile_pickers:
            picker.configure(state="disabled" if busy else "readonly")

    def launch(self, action, label, user_id, **options):
        command = build_command(action, user_id, camera=self.camera.get(), **options)
        if not Path(command[1]).is_file():
            raise ValueError(f"Required script is missing: {command[1]}")
        self.runner.start(command, label)
        self.set_busy(True)
        self.status.set(f"{label} is open. Press Q or Esc in its window to return.")

    def handle(self, callback):
        try:
            callback()
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not start action", str(error), parent=self.window)

    def enroll(self):
        def start():
            user_id = self.new_user.get().strip()
            previous = self.profiles.get(user_id)
            self.launch("enroll", "Enrollment", user_id)
            self.pending_enrollment = (user_id, previous.palm_model_path if previous else None)
        self.handle(start)

    def live(self):
        self.handle(lambda: self.launch("live", "Webcam verification", self.selected_user()))

    def photo(self):
        def choose():
            user_id = self.selected_user(require_enrollment=True)
            path = filedialog.askopenfilename(
                parent=self.window, title="Choose a hand photo",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
            )
            if path:
                self.launch("photo", "Photo verification", user_id, image_path=path)
        self.handle(choose)

    def experiment(self):
        self.handle(lambda: self.launch(
            "experiment", "Experiment", self.selected_user(require_enrollment=True),
            actual_id=self.actual.get().strip(), condition=self.condition.get(),
            distance_cm=self.distance.get(), scale=self.scale.get(),
        ))

    def poll(self):
        result = self.runner.poll()
        if result is not None:
            self.set_busy(False)
            self.refresh()
            pending = self.pending_enrollment
            self.pending_enrollment = None
            if result.returncode == 0:
                self.status.set(f"{result.label} closed.")
                if pending is not None:
                    user_id, previous_path = pending
                    profile = self.profiles.get(user_id)
                    if profile and profile.palm_model_path and profile.palm_model_path != previous_path:
                        self.selected.set(user_id)
                        self.update_summary()
                        self.tabs.select(self.verify_tab)
                        self.status.set(f"Enrollment saved for {user_id}.")
                    else:
                        self.status.set("Enrollment closed before saving a new profile.")
            else:
                self.status.set(f"{result.label} stopped with exit code {result.returncode}. See the log in outputs/launcher.")
                messagebox.showerror(
                    f"{result.label} stopped",
                    f"Exit code: {result.returncode}\n\n{result.detail}\n\nFull log: {result.log_path}",
                    parent=self.window,
                )
        self.window.after(250, self.poll)

    def close(self):
        if self.runner.process is not None:
            messagebox.showinfo("Action still open", "Close the camera or photo window with Q or Esc first.", parent=self.window)
            return
        self.window.destroy()


def main():
    try:
        window = tk.Tk()
        Launcher(window)
        window.mainloop()
    except tk.TclError as error:
        print(f"Could not open the desktop launcher: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
