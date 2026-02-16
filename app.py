# app.py
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

# We use sys.executable so the GUI launches scripts using the SAME Python
# that launched the GUI (your .venv Python), not your global Python.
PY = sys.executable

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Employee Face Recognition")
        self.geometry("420x220")

        self.recognition_proc = None

        tk.Label(self, text="Employee Face Recognition", font=("Segoe UI", 14, "bold")).pack(pady=10)

        self.status = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status).pack(pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        self.btn_register = tk.Button(btn_frame, text="Register New Employee", width=22, command=self.register_employee)
        self.btn_register.grid(row=0, column=0, padx=8, pady=5)

        self.btn_start = tk.Button(btn_frame, text="Start Recognition", width=22, command=self.start_recognition)
        self.btn_start.grid(row=1, column=0, padx=8, pady=5)

        self.btn_stop = tk.Button(btn_frame, text="Stop Recognition", width=22, command=self.stop_recognition, state=tk.DISABLED)
        self.btn_stop.grid(row=2, column=0, padx=8, pady=5)

        # If the user closes the GUI window, we should stop the recognition process cleanly
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_busy(self, busy: bool, msg: str):
        """Enable/disable buttons so the user can't start conflicting actions."""
        self.status.set(msg)
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_register.config(state=state)
        self.btn_start.config(state=state if self.recognition_proc is None else tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL if self.recognition_proc is not None else tk.DISABLED)
        self.update_idletasks()

    def register_employee(self):
        """
        Runs capture_faces.py (collect images) then train_model.py (update encodings).
        We use subprocess.run (blocking) because registration is a "do it now" flow.
        """
        if self.recognition_proc is not None:
            messagebox.showwarning("Recognition running", "Stop recognition before registering.")
            return

        try:
            self.set_busy(True, "Running face capture...")

            # 1) Capture faces
            # This opens the OpenCV window from capture_faces.py and returns when that script exits.
            subprocess.run([PY, "capture_faces.py"], check=True)

            self.set_busy(True, "Training encodings...")

            # 2) Train model / create encodings.pickle
            subprocess.run([PY, "train_model.py"], check=True)

            self.set_busy(False, "Registration complete.")
            messagebox.showinfo("Done", "Registration complete and encodings updated.")

        except subprocess.CalledProcessError as e:
            self.set_busy(False, "Error during registration.")
            messagebox.showerror("Error", f"One of the scripts exited with an error.\n\n{e}")

        except FileNotFoundError:
            self.set_busy(False, "Missing file.")
            messagebox.showerror("Error", "Could not find capture_faces.py or train_model.py in this folder.")

        finally:
            self.set_busy(False, self.status.get())

    def start_recognition(self):
        """
        Starts recognize_and_greet.py as a background process so the GUI stays responsive.
        We use Popen (non-blocking) and keep the process handle to stop it later.
        """
        if self.recognition_proc is not None:
            messagebox.showinfo("Already running", "Recognition is already running.")
            return

        try:
            self.status.set("Starting recognition...")
            self.update_idletasks()

            # Launch recognition in a separate process.
            # This keeps GUI responsive and the OpenCV window belongs to that process.
            self.recognition_proc = subprocess.Popen([PY, "recognize_and_greet.py"])

            self.set_busy(False, "Recognition running.")
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)

        except FileNotFoundError:
            self.recognition_proc = None
            self.set_busy(False, "Missing recognize_and_greet.py")
            messagebox.showerror("Error", "Could not find recognize_and_greet.py in this folder.")

    def stop_recognition(self):
        """Stops the recognition subprocess safely."""
        if self.recognition_proc is None:
            return

        try:
            self.status.set("Stopping recognition...")
            self.update_idletasks()

            self.recognition_proc.terminate()
            self.recognition_proc = None

            self.set_busy(False, "Recognition stopped.")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recognition.\n\n{e}")

    def on_close(self):
        """When GUI closes, terminate recognition if running."""
        if self.recognition_proc is not None:
            try:
                self.recognition_proc.terminate()
            except:
                pass
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
