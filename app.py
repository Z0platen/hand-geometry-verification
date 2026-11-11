"""A small desktop launcher for local hand geometry demonstrations."""

from dataclasses import dataclass
from datetime import datetime
import json
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
    model_path: Path
    sample_count: int | None
    session_count: int


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
        if not models:
            continue
        latest = max(models, key=lambda path: path.parent.name)
        sample_count = None
        try:
            model = json.loads(latest.read_text(encoding="utf-8"))
            count = model.get("enrollment_counts", {}).get(user_id) if isinstance(model, dict) else None
            if type(count) is int and count > 0:
                sample_count = count
        except (OSError, UnicodeError, ValueError, AttributeError):
            # Keep the same latest-session selection as verify.py. Its loader
            # reports malformed models rather than silently selecting an older one.
            pass
        profiles.append(Profile(user_id, latest, sample_count, len(models)))
    return profiles


def build_command(
    action, user_id, *, camera="0", image_path=None, actual_id="",
    condition="standard", root=ROOT, executable=None,
):
    """Construct separate arguments; user-entered values never become shell code."""
    user_id = validate_profile_id(user_id, "New user ID" if action == "enroll" else "Selected profile ID")
    root = Path(root).resolve()
    executable = str(Path(executable or sys.executable).resolve())
    if action == "enroll":
        return [executable, str(root / "enroll.py"), f"--user={user_id}", "--camera", str(camera_index(camera))]
    if action == "live":
        return [executable, str(root / "verify.py"), f"--user={user_id}", "--camera", str(camera_index(camera))]
    if action == "photo":
        if image_path is None or not Path(image_path).is_file():
            raise ValueError("Select an existing image file.")
        return [executable, str(root / "verify.py"), f"--user={user_id}",
                "--image", str(Path(image_path).resolve()), "--show"]
    if action == "experiment":
        return [executable, str(root / "experiment.py"), f"--user={user_id}",
                f"--actual={validate_profile_id(actual_id, 'Actual person ID')}", f"--condition={condition_label(condition)}",
                "--camera", str(camera_index(camera))]
    raise ValueError(f"Unknown action: {action!r}.")


@dataclass(frozen=True)
class RunResult:
    label: str
    returncode: int
    log_path: Path
    detail: str


def process_error_detail(text):
    lines = text.splitlines()
    prefixes = ("Verification error:", "Enrollment error:", "Experiment error:")
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
        self.selected = tk.StringVar()
        self.new_user = tk.StringVar()
        self.camera = tk.StringVar(value="0")
        self.actual = tk.StringVar()
        self.condition = tk.StringVar(value="standard")
        self.summary = tk.StringVar()
        self.status = tk.StringVar(value="Ready. Choose an enrolled profile or create a new one.")
        window.title("Hand Geometry - Project Menu")
        window.minsize(690, 610)
        window.geometry("790x660")
        window.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(window, padding=24)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        ttk.Label(body, text="Hand Geometry Verification", font=("Segoe UI", 20, "bold")).grid(sticky="w")
        ttk.Label(body, text="Enroll a hand, compare a photo, or run a labeled experiment.").grid(sticky="w", pady=(5, 18))

        profiles = ttk.LabelFrame(body, text="Local profiles", padding=14)
        profiles.grid(sticky="ew")
        profiles.columnconfigure(1, weight=1)
        ttk.Label(profiles, text="Selected profile").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.profile_picker = ttk.Combobox(profiles, textvariable=self.selected, state="readonly")
        self.profile_picker.grid(row=0, column=1, sticky="ew")
        self.profile_picker.bind("<<ComboboxSelected>>", lambda _event: self.update_summary())
        refresh = ttk.Button(profiles, text="Refresh", command=self.refresh)
        refresh.grid(row=0, column=2, padx=(10, 0))
        self.busy_controls.append(refresh)
        ttk.Label(profiles, textvariable=self.summary, wraplength=650).grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        create = ttk.Frame(body)
        create.grid(sticky="ew", pady=(16, 8))
        ttk.Label(create, text="New user ID").pack(side="left", padx=(0, 12))
        new_entry = ttk.Entry(create, textvariable=self.new_user, width=22)
        new_entry.pack(side="left", padx=(0, 12))
        enroll = ttk.Button(create, text="Enroll new user", command=self.enroll)
        enroll.pack(side="left")
        self.busy_controls.extend([new_entry, enroll])
        ttk.Label(body, text="Enrollment saves 10 manually requested hand photos locally.").grid(sticky="w", pady=(0, 14))

        camera_row = ttk.Frame(body)
        camera_row.grid(sticky="ew", pady=(0, 14))
        ttk.Label(camera_row, text="Camera index").pack(side="left", padx=(0, 12))
        camera_entry = ttk.Entry(camera_row, textvariable=self.camera, width=6)
        camera_entry.pack(side="left")
        ttk.Label(camera_row, text="Usually 0; try 1 for another camera.").pack(side="left", padx=12)
        self.busy_controls.append(camera_entry)

        actions = ttk.Frame(body)
        actions.grid(sticky="ew", pady=(0, 18))
        for label, callback in (("Verify with webcam", self.live), ("Verify a photo", self.photo)):
            button = ttk.Button(actions, text=label, command=callback)
            button.pack(side="left", padx=(0, 12))
            self.busy_controls.append(button)

        experiment = ttk.LabelFrame(body, text="Manual experiment", padding=14)
        experiment.grid(sticky="ew")
        experiment.columnconfigure(1, weight=1)
        ttk.Label(experiment, text="Actual person ID").grid(row=0, column=0, sticky="w", padx=(0, 12))
        actual_entry = ttk.Entry(experiment, textvariable=self.actual)
        actual_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(experiment, text="Condition").grid(row=1, column=0, sticky="w", pady=(10, 0))
        condition_entry = ttk.Entry(experiment, textvariable=self.condition)
        condition_entry.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        experiment_button = ttk.Button(experiment, text="Run experiment", command=self.experiment)
        experiment_button.grid(row=0, column=2, rowspan=2, padx=(14, 0))
        ttk.Label(experiment, text="Enter the ID of the person showing their hand. For your own test, use the selected profile's ID.", wraplength=610).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))
        use_selected = ttk.Button(experiment, text="Use selected profile ID", command=self.use_selected_actual)
        use_selected.grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.busy_controls.extend([actual_entry, condition_entry, experiment_button, use_selected])

        ttk.Label(body, text="Research demo: dataset threshold; webcam accuracy is not calibrated.", wraplength=700).grid(sticky="w", pady=(18, 10))
        ttk.Separator(body).grid(sticky="ew")
        ttk.Label(body, textvariable=self.status, wraplength=700).grid(sticky="w", pady=(12, 0))
        self.refresh()
        window.after(250, self.poll)

    def refresh(self):
        try:
            self.profiles = {profile.user_id: profile for profile in list_profiles(ROOT / "captures")}
        except OSError as error:
            messagebox.showerror("Could not list profiles", str(error), parent=self.window)
            return
        self.profile_picker["values"] = tuple(self.profiles)
        if self.selected.get() not in self.profiles:
            self.selected.set(next(iter(self.profiles), ""))
        self.update_summary()

    def update_summary(self):
        profile = self.profiles.get(self.selected.get())
        if profile is None:
            self.summary.set("No completed profiles yet. Enter a new user ID below to enroll.")
        else:
            count = f"{profile.sample_count} samples" if profile.sample_count is not None else "sample count unavailable"
            self.summary.set(f"{len(self.profiles)} completed profile(s)  |  Selected: {profile.user_id}  |  Latest enrollment: {count}")

    def selected_user(self):
        user_id = self.selected.get()
        if user_id not in self.profiles:
            raise ValueError("Enroll a user first, then choose their profile.")
        return user_id

    def set_busy(self, busy):
        for widget in self.busy_controls:
            widget.configure(state="disabled" if busy else "normal")
        self.profile_picker.configure(state="disabled" if busy else "readonly")

    def launch(self, action, label, user_id, **options):
        command = build_command(action, user_id, camera=self.camera.get(), **options)
        if not Path(command[1]).is_file():
            raise ValueError(f"Required script is missing: {command[1]}")
        self.runner.start(command, label)
        self.set_busy(True)
        self.status.set(f"{label} is open. Close its window with Q or Esc to return.")

    def handle(self, callback):
        try:
            callback()
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not start action", str(error), parent=self.window)

    def enroll(self):
        self.handle(lambda: self.launch("enroll", "Enrollment", self.new_user.get().strip()))

    def live(self):
        self.handle(lambda: self.launch("live", "Webcam verification", self.selected_user()))

    def photo(self):
        def choose():
            user_id = self.selected_user()
            path = filedialog.askopenfilename(
                parent=self.window, title="Choose a hand photo",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")],
            )
            if path:
                self.launch("photo", "Photo verification", user_id, image_path=path)
        self.handle(choose)

    def use_selected_actual(self):
        self.handle(lambda: self.actual.set(self.selected_user()))

    def experiment(self):
        self.handle(lambda: self.launch(
            "experiment", "Manual experiment", self.selected_user(),
            actual_id=self.actual.get().strip(), condition=self.condition.get(),
        ))

    def poll(self):
        result = self.runner.poll()
        if result is not None:
            self.set_busy(False)
            self.refresh()
            if result.returncode == 0:
                self.status.set(f"{result.label} closed. Ready for the next action.")
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
