from backend.app.ingestion.scanner import RepositoryScanner


def main():
    scanner = RepositoryScanner(".")

    inventory = scanner.scan()

    print(f"Total files scanned: {len(inventory)}\n")

    for file in inventory:
        print(file)


if __name__ == "__main__":
    main()