
from .cmd import mic
from .ui_cmd import ui_mic
import pandas as pd
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from Qt.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QLineEdit, QHBoxLayout,
    QListWidget, QListWidgetItem
)
from chimerax.atomic import AtomicStructure


class MICTool(ToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:docs/user/commands/tutorial.html"

    def __init__(self, session, tool_name):
        session.logger.info("Initializing MICTool...")  
        super().__init__(session, tool_name)
        self.display_name = "Metric Ion Classification"

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Model Selection Section
        instruction = QLabel("Select one or more PDB models to run MIC on.\n(Leave empty to run on all models.)")
        layout.addWidget(instruction)

        model_label = QLabel("Model to Run MIC On")
        layout.addWidget(model_label)

        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.MultiSelection)

        # Add one item for each model
        for model in self.session.models.list():
            if model.name.lower().endswith(".pdb"):
                item = QListWidgetItem(model.name)
                self.model_list.addItem(item)

        layout.addWidget(self.model_list)

        # FingerPrint Type Dropdown
        fingerprint_label = QLabel("Fingerprint Type")
        layout.addWidget(fingerprint_label)
        self.fingerprint_type_combo = QComboBox()
        self.fingerprint_type_combo.addItems(['prune-eifp', 'prune-fifp', 'non-prune-eifp', 'non-prune-fifp'])
        layout.addWidget(self.fingerprint_type_combo)

        # Ion Classes Considered Dropdown
        ion_class_label = QLabel("Ion Classes Considered")
        layout.addWidget(ion_class_label)
        self.ion_class_combo = QComboBox()
        self.ion_class_combo.addItems(['prevalent set', 'extended set'])
        layout.addWidget(self.ion_class_combo)

        # Ions of Interest Section with Dropdown and LineEdit for Custom Input
        ions_label = QLabel("Ions of Interest")
        layout.addWidget(ions_label)

        self.ions_combo = QComboBox()
        self.ions_combo.addItems(['all ions & solvent', 'all ions', 'solvent', 'custom...'])
        layout.addWidget(self.ions_combo)

        self.ions_input = QLineEdit()
        self.ions_input.setPlaceholderText("e.g. @CA; e.g. '@CL | @MN')")
        self.ions_input.setStyleSheet("color: gray;")
        self.ions_input.setVisible(False)
        self.ions_input.textChanged.connect(self.on_ions_input_change)
        layout.addWidget(self.ions_input)

        self.ions_combo.currentTextChanged.connect(self.on_ions_combo_change)

        # Run MIC Button
        self.run_button = QPushButton("Run MIC", self.tool_window.ui_area)
        self.run_button.clicked.connect(self.run_mic_command)
        layout.addWidget(self.run_button)

        # Display Results Table
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        # Handle row-clicking: zoom in corresponding ion/water in session
        # self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.cellClicked.connect(self.handle_table_click)
        self.session.logger.info("Connected cellClicked to handle_table_click")

        # Download Table
        self.download_button = QPushButton("Download Results", self.tool_window.ui_area)
        self.download_button.clicked.connect(self.download_results)
        self.download_button.setEnabled(False)  
        layout.addWidget(self.download_button)

        # Auto-refresh model list when models are added or removed in ChimeraX
        # self.session.triggers.add_handler('models changed', self.refresh_model_list)
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.manage(None)


    def refresh_model_list(self, *args):
        self.model_list.clear()
        for model in self.session.models.list():
            if model.name.lower().endswith(".pdb"):
                self.model_list.addItem(model.name)

    def update_table(self, results_df):
        # Assuming results_df is already a DataFrame passed from run_mic_command
        self.results_df = results_df

        self.table_widget.setColumnCount(len(self.results_df.columns))
        self.table_widget.setRowCount(len(self.results_df))

        # Setting the column headers to match the DataFrame columns
        self.table_widget.setHorizontalHeaderLabels(self.results_df.columns.tolist())

        # Populating the table with DataFrame content
        for index, row in self.results_df.iterrows():
            for col_index, (column, value) in enumerate(row.items()):
                if pd.isna(value):
                    item_text = QTableWidgetItem("N/A")
                elif isinstance(value, float):
                    item_text = QTableWidgetItem(f"{value:.4f}")
                else:
                    item_text = QTableWidgetItem(str(value))
                self.table_widget.setItem(index, col_index, item_text)

        # Adjusting the column widths to fit the content
        self.table_widget.resizeColumnsToContents()

    def handle_table_click(self, row, column):
        if not hasattr(self, 'results_df'):
            self.session.logger.warning("Results not available yet. Click ignored.")
            return
    
        self.session.logger.info(f"Row clicked: {row}, column: {column}")
        try:
            entry = self.results_df.iloc[row]['Entry']
            pdb_name = self.results_df.iloc[row]['PDB File']

            # Get model that matches the PDB file
            base_model = next((m for m in self.session.models.list() if m.name == pdb_name), None)
            if not base_model:
                self.session.logger.error(f"Base model not found for PDB file: {pdb_name}")
                return

            # Look for submodel #~.1 that contains the ligands
            sub_model_id = f"{base_model.id_string}.1"
            sub_model = next((m for m in self.session.models.list() if m.id_string == sub_model_id), None)
            if not sub_model:
                self.session.logger.error(f"Submodel {sub_model_id} not found. Check whether MIC has been successfully run or not")
                return

            # Use proper ChimeraX atom spec: E.g. A:MG:201
            chain, resname, resnum = entry.split(":")
            atom_spec = f"#{sub_model_id}/{chain}:{resnum}"  # precise residue selection

            # Select and zoom with 'view orient'
            # 1. Hide all models except the ion submodel
            for model in self.session.models.list():
                if model.id_string.startswith(sub_model_id) or model.id_string == base_model.id_string:
                    model.display = True  # show submodel + parent
                else:
                    model.display = False  # hide unrelated models
            # 2. Select and orient to the ion
            run(self.session, f"select {atom_spec}")
            run(self.session, f"show {atom_spec}")  # force shown if not already
            run(self.session, "view sel clip false pad 0.5")
            # 3. Restore visibility of all models
            for model in self.session.models.list():
                model.display = True
            self.session.logger.info(f"Selected, oriented, and restored view for: {atom_spec}")

        except Exception as e:
            self.session.logger.error(f"Zoom failed: {str(e)}")


    def download_results(self):
        file_name, _ = QFileDialog.getSaveFileName(self.tool_window.ui_area, "Save Results", "", "CSV Files (*.csv)")
        if file_name:
            if not file_name.endswith('.csv'):
                file_name += '.csv'
            self.results_df.to_csv(file_name) #, index=False
            self.session.logger.info(f"Results saved to {file_name}")


    def color_ions_by_element(self):
        try:
            run(self.session, "color ions byelement")
            self.session.logger.info("Ions need to be colored by element.")
        except Exception as e:
            self.session.logger.error(f"Failed to color ions by element: {str(e)}")

    def on_ions_combo_change(self, text):
        if text == 'custom...':
            self.ions_input.setVisible(True)
            self.ions_input.setPlaceholderText("e.g. @CA; e.g. '@CL | @MN'")
            if not self.ions_input.text():
                self.ions_input.setStyleSheet("color: gray;")
            else:
                self.ions_input.setStyleSheet("color: black;")
        else:
            self.ions_input.setVisible(False)
            self.ions_input.clear()

    def on_ions_input_change(self):
        if self.ions_input.text():
            self.ions_input.setStyleSheet("color: black;")
        else:
            self.ions_input.setStyleSheet("color: gray;")

    def run_mic_command(self):

        if True:
            self.session.logger.info("Run MIC command triggered")  
            fingerprint_type = self.fingerprint_type_combo.currentText()
            ion_class = self.ion_class_combo.currentText()

            combined_results = []
            error_messages = []

            all_models = models_to_run = [m for m in self.session.models.list() if isinstance(m, AtomicStructure)]
            selected_names = [item.text() for item in self.model_list.selectedItems()]

            # If none selected, default to all
            if not selected_names:
                models_to_run = all_models
            else:
                models_to_run = [m for m in self.session.models.list() if m.name in selected_names and isinstance(m, AtomicStructure)]

            if not models_to_run:
                self.session.logger.error("No models selected or found in session.")
                return

            for model in models_to_run:
                pdb_model_name = model.name
                model_id = model.id_string
                self.session.logger.info(f"Running MIC on model: {pdb_model_name} ({model_id})")

                selected_ions = self.ions_combo.currentText()
                if selected_ions == 'custom...':
                    ions_argument = self.ions_input.text().strip()
                    if (ions_argument.startswith('"') and ions_argument.endswith('"')) or \
                        (ions_argument.startswith("'") and ions_argument.endswith("'")):
                        ions_argument = ions_argument[1:-1]
                elif selected_ions == 'all ions & solvent':
                    ions_argument = None  # Default to selecting all ions and solvent
                elif selected_ions == 'all ions':
                    ions_argument = "ions"
                elif selected_ions == 'solvent':
                    ions_argument = "solvent"

                self.session.logger.info(f"Fingerprint type: {fingerprint_type}, Ion class: {ion_class}, Model: {pdb_model_name}, Model ID: {model_id}")  

                try:
                    # Run the mic command on the correct model
                    results = ui_mic(self.session, model_id, ions_argument)
                    if results is None:
                        error_messages.append(f"No ions selected in {pdb_model_name} (model {model_id}).")
                        continue

                    self.session.logger.info(results)
                    
                    # Create DataFrame from results
                    from io import StringIO
                    df = pd.read_csv(StringIO(results), delim_whitespace=True, keep_default_na=False, na_values=[''])
                    
                    # Reset index to create 'Entry' column
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'Entry'}, inplace=True)

                    # Add 'PDB File' column
                    df['PDB File'] = pdb_model_name

                    combined_results.append(df)

                    self.color_ions_by_element()
                    
                except Exception as e:
                    error_messages.append(f"Failed to run MIC command on {pdb_model_name}: {str(e)}")
                            
            if combined_results:
                # Combine all results into one DataFrame
                final_results = pd.concat(combined_results, ignore_index=True)

                # Reorder columns to have 'PDB File' as the first column
                columns = ['PDB File'] + [col for col in final_results.columns if col != 'PDB File']
                final_results = final_results[columns]

                self.update_table(final_results)  # Pass the DataFrame directly to update_table
                self.download_button.setEnabled(True)
            else:
                error_messages.append("No results were generated.")

            if error_messages:
                # Consolidate errors into a single message
                consolidated_error_message = "\n".join(error_messages)
                final_error_message = f"{consolidated_error_message}\nMake sure the PDB file of interest is uploaded.\nMake sure the ions of interest exist in your PDB file."

                # Log the error without raising an exception to avoid multiple error windows
                self.session.logger.error(final_error_message)



