import os

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

notebook_list = ["indices.ipynb"]

for notebook_filename in notebook_list:

    notebook_path = os.path.abspath(os.path.join("../notebooks", notebook_filename))

    try:
        with open(notebook_path) as f:
            notebook_content = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        ep.preprocess(notebook_content, {"metadata": {"path": os.path.abspath("../notebooks")}})

    except CellExecutionError as e:
        pytest.fail(f"Notebook {notebook_filename} failed during execution: {str(e)}")
    except Exception as e:
        pytest.fail(f"Notebook {notebook_filename} failed with error: {str(e)}")
