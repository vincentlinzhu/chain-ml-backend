def run_tests(qa):
    print(qa.invoke({"question": "Which country runs the world's greatest trading surplus, and which country runs the greatest deficit?"})['answer'])
    print(qa.invoke({"question": "Which country runs the world's greatest trading surplus, and which country runs the greatest deficit?"})['source_documents'])