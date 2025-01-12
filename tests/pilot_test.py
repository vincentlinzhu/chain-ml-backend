def run_tests(qa):
    ## Answered in document 101
    print(qa.invoke({"question": "How old was the pilot who crashed the flight in Old Bridge NJ?"})['answer'])
    print(qa.invoke({"question": "How old was the pilot who crashed the flight in Old Bridge NJ?"})['source_documents'])

    ## Answered in document 25
    print(qa.invoke({"question": "For the crash with the Registration number N747PK, what caused the fire?"})['answer'])

    ## Answered in document 27
    print(qa.invoke({"question": "Where was the digital flight recorder shipped to for the flight that crashed on January 23, 2023?"})['answer'])

    ## Answered in document 42
    print(qa.invoke({"question": "For the Cessna 172M that crashed in Skull Valley, how long had it been since the last FAA medical exam?"})['answer'])