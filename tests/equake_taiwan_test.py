def run_tests_taiwan(qa):
    print(qa.invoke({"question": "Is United Microelectronics Corp the largest chipmaker? If not, what rank are they?"})['answer'])
    print(qa.invoke({"question": "Is United Microelectronics Corp the largest chipmaker? If not, what rank are they?"})['source_documents'])