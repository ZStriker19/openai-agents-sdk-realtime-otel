import json
import subprocess
import tempfile
from pathlib import Path


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.
    """
    return json.dumps({"city": city, "temperature": "72°F", "condition": "sunny"})


def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Args:
        expression: A math expression to evaluate, e.g. '2 + 2'.
    """
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def run_python_code(code: str) -> str:
    """Write and execute a Python script, returning its stdout/stderr.

    Args:
        code: The Python source code to execute.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=tempfile.gettempdir(), delete=False
    ) as f:
        f.write(code)
        script_path = Path(f.name)

    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: script timed out after 30 seconds"
    finally:
        script_path.unlink(missing_ok=True)


async def write_file(file_path: str, content: str) -> str:
    """Write content to a file on disk.

    Args:
        file_path: The path to write the file to.
        content: The content to write into the file.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"
