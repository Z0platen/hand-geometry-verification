"""Read-only native viewer for grouped palm verification trials."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def _number(value):
    return f"{value:g}" if value is not None else "n/a"


def _rate(rate, numerator, denominator):
    if rate is None:
        return f"n/a ({numerator}/{denominator})"
    return f"{rate:.1%} ({numerator}/{denominator})"


def group_row_values(group):
    """Format one compiler group while retaining all conditional denominators."""
    failures = group["acquisition_failures"]
    attempts = group["attempts"]
    return (
        ", ".join(group["claimed_ids"]) or "none",
        group["condition"],
        _number(group["distance_cm"]),
        _number(group["input_scale"]),
        str(attempts),
        _rate(group["fmr"], group["false_accepts"], group["impostor_comparisons"]),
        _rate(group["fnmr"], group["false_rejects"], group["genuine_comparisons"]),
        _rate(group["acquisition_failure_rate"], failures, attempts),
    )


def group_details(group):
    """Build selectable per-group detail text without truncating model identity."""
    lines = [
        f"Condition: {group['condition']}",
        f"Distance (cm): {_number(group['distance_cm'])}",
        f"Input scale: {_number(group['input_scale'])}",
        f"Runs: {group['runs']}",
        f"Model SHA-256: {group['model_sha256']}",
        f"Thresholds: {group['thresholds']}",
        f"Run IDs: {', '.join(group['run_ids']) or 'none'}",
        f"Claimed profile IDs: {', '.join(group['claimed_ids']) or 'none'}",
        f"Actual person IDs: {', '.join(group['actual_ids']) or 'none'}",
        f"Attempts: {group['attempts']} total; {group['genuine_attempts']} genuine; {group['impostor_attempts']} impostor",
        f"Valid acquisitions: {group['valid_acquisitions']}; acquisition failures: {group['acquisition_failures']}",
        f"Genuine comparisons: {group['genuine_comparisons']}; false rejects: {group['false_rejects']}",
        f"Impostor comparisons: {group['impostor_comparisons']}; false accepts: {group['false_accepts']}",
        f"FMR: {_rate(group['fmr'], group['false_accepts'], group['impostor_comparisons'])}",
        f"FNMR: {_rate(group['fnmr'], group['false_rejects'], group['genuine_comparisons'])}",
        f"Acquisition failure rate: {_rate(group['acquisition_failure_rate'], group['acquisition_failures'], group['attempts'])}",
        f"Coverage: {_rate(group['coverage'], group['valid_acquisitions'], group['attempts'])}",
    ]
    return "\n".join(lines)


def report_intro(report):
    runs_seen = report["runs_seen"]
    valid_runs = report["valid_runs"]
    skipped_runs = report["skipped_runs"]
    intro = f"Runs seen: {runs_seen}    Valid runs: {valid_runs}    Skipped runs: {skipped_runs}"
    if runs_seen == 0:
        return intro + (
            "\nNo trial runs yet. In Experiments, select a profile and Actual person ID, choose Run experiment, "
            "press T to save attempts, then Q to close and select View results."
        )
    if valid_runs == 0:
        return intro + "\nNo complete valid runs to report. Review the warnings below."
    if not report["groups"]:
        return intro + "\nNo report groups are available. Review the warnings below."
    return intro


class PalmResultsWindow(tk.Toplevel):
    """Scrollable report window; compiler output is read-only and grouped as supplied."""

    COLUMNS = (
        ("profiles", "Profile IDs", 165),
        ("condition", "Condition", 115),
        ("distance", "Distance cm", 75),
        ("scale", "Scale", 65),
        ("attempts", "Attempts", 65),
        ("fmr", "FMR (FA/valid I)", 130),
        ("fnmr", "FNMR (FR/valid G)", 140),
        ("failures", "Acq. fail (fail/all)", 125),
    )

    def __init__(self, parent, report_root):
        super().__init__(parent)
        self.report_root = Path(report_root)
        self.report = None
        self.group_by_iid = {}
        self.title("Combined verification results")
        self.geometry("1120x680")
        self.minsize(900, 520)
        self.transient(parent)

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(body)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(toolbar, text="Save HTML…", command=self.save_html).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Close", command=self.destroy).pack(side="right")

        self.summary = tk.StringVar(value="Loading trial results…")
        ttk.Label(body, textvariable=self.summary, justify="left", wraplength=850).grid(
            row=1, column=0, sticky="ew", pady=(0, 8),
        )

        table_frame = ttk.Frame(body)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        columns = tuple(column[0] for column in self.COLUMNS)
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        for key, heading, width in self.COLUMNS:
            self.tree.heading(key, text=heading)
            self.tree.column(key, width=width, minwidth=65, stretch=False, anchor="w")
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self.show_selected_group)

        details_pane = ttk.Panedwindow(body, orient="horizontal")
        details_pane.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        detail_frame = ttk.LabelFrame(details_pane, text="Selected group details", padding=6)
        warning_frame = ttk.LabelFrame(details_pane, text="Warnings", padding=6)
        details_pane.add(detail_frame, weight=3)
        details_pane.add(warning_frame, weight=2)
        self.details = self._scroll_text(detail_frame)
        self.warnings = self._scroll_text(warning_frame)
        body.rowconfigure(3, weight=1)
        self.status = tk.StringVar(value="")
        ttk.Label(body, textvariable=self.status, anchor="w").grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.refresh()

    @staticmethod
    def _scroll_text(parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        widget = tk.Text(frame, wrap="word", height=7, state="disabled")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scrollbar.set)
        widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return widget

    @staticmethod
    def _set_text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def refresh(self):
        from palm_reports import compile_reports

        self.status.set("")
        try:
            report = compile_reports(self.report_root)
        except (OSError, ValueError, TypeError) as error:
            self.report = None
            self.summary.set(f"Could not read trial reports: {error}")
            self._set_text(self.details, "")
            self._set_text(self.warnings, str(error))
            self.tree.delete(*self.tree.get_children())
            self.group_by_iid.clear()
            return
        self.report = report
        self.summary.set(
            report_intro(report)
            + "\nFMR = false accepts / valid impostor comparisons; FNMR = false rejects / valid genuine comparisons. "
            "Acquisition failures are excluded from those denominators."
        )
        warnings = report["warnings"]
        self._set_text(self.warnings, "\n".join(warnings) if warnings else "No warnings.")
        self._set_text(self.details, "Select a group to inspect its IDs, model hash, thresholds, and denominators.")
        self.tree.delete(*self.tree.get_children())
        self.group_by_iid.clear()
        for index, group in enumerate(report["groups"]):
            iid = str(index)
            self.group_by_iid[iid] = group
            self.tree.insert("", "end", iid=iid, values=group_row_values(group))
        if report["groups"]:
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.show_selected_group()

    def show_selected_group(self, _event=None):
        selection = self.tree.selection()
        if selection:
            self._set_text(self.details, group_details(self.group_by_iid[selection[0]]))

    def save_html(self):
        if self.report is None:
            messagebox.showerror("Cannot save report", "Refresh a valid report before exporting.", parent=self)
            return
        from palm_reports import write_html_report

        reports_directory = self.report_root.parent / "reports"
        try:
            reports_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            messagebox.showerror("Cannot save report", str(error), parent=self)
            return
        destination = filedialog.asksaveasfilename(
            parent=self, title="Save verification report", initialdir=str(reports_directory),
            initialfile="combined_verification_report.html",
            filetypes=[("HTML report", "*.html"), ("All files", "*.*")], defaultextension=".html",
        )
        if not destination:
            return
        try:
            saved_path = write_html_report(self.report, destination)
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror("Cannot save report", str(error), parent=self)
            return
        self.status.set(f"Saved HTML report: {saved_path}")
