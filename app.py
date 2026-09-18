from input_handler import handle_input
from model1.security_gateway import security_check, display_result


def main():

    print("\n" + "=" * 70)
    print("PROMPT INJECTION DEFENSE SYSTEM")
    print("=" * 70)

    print("\n1. Text Prompt")
    print("2. PDF")
    print("3. Exit")

    choice = input("\nChoose input type: ").strip()

    if choice == "3":
        return

    if choice == "1":

        text = input("\nEnter your prompt: ").strip()

        processed_input = handle_input(
            "text",
            text
        )

    elif choice == "2":

        pdf_path = input(
            "\nEnter PDF path: "
        ).strip()

        processed_input = handle_input(
            "pdf",
            pdf_path
        )

    else:

        print("\nInvalid choice.")
        return

    result = security_check(
        processed_input
    )

    display_result(result)


if __name__ == "__main__":
    main()