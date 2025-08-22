import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from tkinter import filedialog, messagebox
from src.iohandler import *
from src.grouping import *

class AGAT:
    def __init__(self, root):
        self.root = root
        self.root.title("AGAT")

        # Variables to store file paths
        self.config_path = None
        self.users_path = None

        # Instructions
        instructions = (
            "1. Select the two input files\n"
            "2. Click 'Generate Grouping'\n"
            "3. Save the resulting file\n"
            "\n"
            "Need help? Click 'Show Examples' for input examples."
        )
        tk.Label(root, text=instructions, justify="left", padx=10, pady=10).pack()

        # Example Button
        tk.Button(root, text="Show Examples", command=self.show_examples).pack(pady=5)

        # File selection buttons
        tk.Button(root, text="Select Config File (JSON)", command=self.select_config_file).pack(pady=5)
        self.config_label = tk.Label(root, text="No file selected", fg="gray")
        self.config_label.pack()

        tk.Button(root, text="Select Users File (CSV)", command=self.select_users_file).pack(pady=5)
        self.users_file_label = tk.Label(root, text="No file selected", fg="gray")
        self.users_file_label.pack()

        # Generate Grouping Button
        tk.Button(root, text="Generate Grouping", command=self.run_tool).pack(pady=10)


    def show_examples(self):
        # Example text
        example_text = (
            "CONFIG FILE (JSON):\n"
            "{\n"
            '    "preferences": [\n'
            '        {\n'
            '            "name": "Experience",\n'
            '            "matching_type": "homogeneous",\n'
            '            "weight": 0.6,\n'
            '            "value_type": "numerical",\n'
            '            "min_value": 1,\n'
            '            "max_value": 10\n'
            '        },\n'
            '        {\n'
            '            "name": "Team-Role",\n'
            '            "matching_type": "heterogeneous",\n'
            '            "weight": 0.4,\n'
            '            "value_type": "categorical",\n'
            '            "categories": ["Analyst", "Developer", "Designer"]\n'
            "        }\n"
            "    ],\n"
            '    "group_size": 3,\n'
            '    "linkage_method": "UPGMA",\n'
            '    "repair_strategy": "merge"\n'
            "}\n\n\n"
            "USERS FILE (CSV):\n"
            "user_id, user_name, Experience, Team-Role\n"
            "1, Alice, 2, Analyst\n"
            "2, Bob, 9, Developer\n"
            "3, Charlie, 4, Analyst\n"
            "4, David, 6, Developer\n"
            "5, Elvis, 8, Designer\n"
            "6, Fiona, 2, Designer\n\n\n"
            "OPTIONS FOR CONFIGURATION:\n"
            "  - matching_type: ['heterogeneous', 'homogeneous']\n"
            "  - value_type: ['numerical', 'categorical']\n"
            "  - linkage_method: ['single', 'complete', 'UPGMA', 'WPGMA', 'total']\n"
            "  - repair_strategy: ['merge', 'break']\n"
        )
        messagebox.showinfo("Input Examples", example_text)


    def select_config_file(self):
        # Select the Config JSON file
        filepath = filedialog.askopenfilename(
            title="Select Config JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            self.config_path = filepath
            self.config_label.config(text=f"Selected: {filepath}", fg="black")


    def select_users_file(self):
        # Select the users CSV file
        filepath = filedialog.askopenfilename(
            title="Select Users CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filepath:
            self.users_path = filepath
            self.users_file_label.config(text=f"Selected: {filepath}", fg="black")


    def run_tool(self):
        # Ensure both files are selected
        if not self.config_path or not self.users_path:
            messagebox.showerror("Error", "Please select both input files before running the tool.")
            return

        try:
            # Step 1: Create the pool
            pool = create_pool_from_files(self.config_path, self.users_path)
            
            # Step 2: Generate the grouping
            group_size, linkage_method, repair_strategy = get_algorithm_params(self.config_path)
            grouping = generate_grouping(pool, group_size, linkage_method, repair_strategy)

            # Step 3: Save the output
            output_file = filedialog.asksaveasfilename(
                title="Save Grouping Results",
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                initialfile="generated_grouping.json"
            )
            if output_file:
                create_output_file(grouping, pool, output_file)
                messagebox.showinfo("Success", f"Results saved to {output_file}")
            else:
                messagebox.showwarning("Warning", "No output file selected. Results were not saved.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            raise e


# Run the app
root = tk.Tk()
app = AGAT(root)
root.mainloop()


