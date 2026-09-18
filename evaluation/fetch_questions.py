from evaluation.client import fetch_questions


if __name__ == "__main__":
    for question in fetch_questions():
        print(question)
