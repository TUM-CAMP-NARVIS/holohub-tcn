# main.py
import os
import slint

MODULE_BASE_PATH = os.path.dirname(__file__)
UI_BASE_PATH = os.path.join(MODULE_BASE_PATH, "ui")
SLINT_LIBRARY_PATHS = {"sleek-ui": os.path.join(MODULE_BASE_PATH, "modules", "sleek-ui")}


# ---------------------------------------------------------------------------
# 1. Example backend with getters/setters
# ---------------------------------------------------------------------------

class ParameterBackend:
    """
    Example backend: stores values in a dict and does very simple type checks.
    Replace this with your actual backend.
    """
    def __init__(self, schema):
        # schema: list of (name, type, initial_value)
        self._types = {}
        self._values = {}

        for name, typ, initial in schema:
            self._types[name] = typ
            self._values[name] = initial

    def get_schema(self):
        """Return normalized schema including current value."""
        result = []
        for name, typ in self._types.items():
            result.append((name, typ, self._values[name]))
        return result

    def get(self, name):
        return self._values[name]

    def set(self, name, value):
        typ = self._types[name]

        if typ == "int":
            if not isinstance(value, int):
                raise ValueError(f"{name} must be int")
        elif typ == "float":
            if not isinstance(value, float):
                raise ValueError(f"{name} must be float")
        elif typ == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be bool")
        elif typ == "string":
            if not isinstance(value, str):
                raise ValueError(f"{name} must be string")
        else:
            raise ValueError(f"Unknown type {typ!r}")

        self._values[name] = value
        print(f"Backend: {name} = {value!r}")


# ---------------------------------------------------------------------------
# 2. Slint-based parameter editor
# ---------------------------------------------------------------------------

class ParameterEditorApp(slint.load_file(os.path.join(UI_BASE_PATH, "parameter_editor.slint"),
                                         library_paths=SLINT_LIBRARY_PATHS).ParameterEditor):
    def __init__(self, backend: ParameterBackend):
        super().__init__()

        self._backend = backend

        # Build Parameter[] model from backend schema
        parameter_rows = []

        for name, typ, initial in backend.get_schema():
            row = {
                "name": name,
                "type": typ,
                "int_value": 0,
                "float_value": 0.0,
                "bool_value": False,
                "string_value": "",
            }

            # Initialize typed fields + displayed text
            if typ == "int":
                initial_int = int(initial)
                row["int_value"] = initial_int
                row["string_value"] = str(initial_int)  # for table view
            elif typ == "float":
                initial_float = float(initial)
                row["float_value"] = initial_float
                row["string_value"] = str(initial_float)
            elif typ == "bool":
                initial_bool = bool(initial)
                row["bool_value"] = initial_bool
                row["string_value"] = "true" if initial_bool else "false"
            elif typ == "string":
                row["string_value"] = str(initial)
            else:
                raise ValueError(f"Unsupported parameter type: {typ!r}")

            parameter_rows.append(row)

        # Expose models to Slint
        self.parameters = slint.ListModel(parameter_rows)

        self.validation_error = ""

    # Bind the Slint callback `parameter-changed` to this Python method
    @slint.callback(name="parameter-changed")
    def on_parameter_changed(
        self,
        index: int,
        name: str,
        typ: str,
        int_value: int,
        float_value: float,
        bool_value: bool,
        string_value: str,
    ):
        """
        Called by the Slint UI whenever the user edits a parameter.
        We validate & update backend and keep the table model in sync.
        """

        # Decide which value field actually matters
        try:
            if typ == "int":
                new_value = int(int_value)

            elif typ == "float":
                # For float we get the text from string_value and parse it here
                if string_value.strip() == "":
                    raise ValueError("Float value may not be empty")
                new_value = float(string_value)

            elif typ == "bool":
                new_value = bool(bool_value)

            elif typ == "string":
                new_value = string_value
                display_text = string_value
            else:
                raise ValueError(f"Unsupported type {typ!r}")

            # Push to backend (may raise ValueError)
            self._backend.set(name, new_value)

            # Clear error
            self.validation_error = ""

        except ValueError as e:
            # Reject change: restore old value from backend
            old_value = self._backend.get(name)
            self.validation_error = f"{name}: {e}"

            # Repair the parameters model so UI snaps back
            param = dict(self.parameters[index])
            if typ == "int":
                param["int_value"] = int(old_value)
                param["string_value"] = str(old_value)
            elif typ == "float":
                param["float_value"] = float(old_value)
                param["string_value"] = str(old_value)
            elif typ == "bool":
                param["bool_value"] = bool(old_value)
                param["string_value"] = "true" if old_value else "false"
            elif typ == "string":
                param["string_value"] = str(old_value)

            self.parameters[index] = param

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Your schema: (name, type, initial_value)
    schema = [
        ("iterations", "int", 10),
        ("learning_rate", "float", 0.01),
        ("use_gpu", "bool", True),
        ("output_dir", "string", "/tmp/output"),
    ]

    backend = ParameterBackend(schema)
    app = ParameterEditorApp(backend)
    app.run()
