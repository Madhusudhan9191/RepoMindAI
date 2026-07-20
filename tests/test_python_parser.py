from backend.app.parsers.python_parser import PythonParser


def main():
    parser = PythonParser("backend/app/sample.py")

    metadata = parser.extract_metadata()

    print(metadata)


if __name__ == "__main__":
    main()