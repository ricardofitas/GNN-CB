"""Build the current expanded report; pilot builder remains separately available."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('build_expanded_report.py')),run_name='__main__')
