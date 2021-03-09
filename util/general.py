# General util file. 
# Used for storing very basic functions used throughout the project.

def print_result(succeeded=True):

    if succeeded:
        print("\033[92mOK\033[0m")
    elif not succeeded:
        print("\033[91mFAIL\033[0m")
    else:
        raise ValueError("Parameter 'succeeded' should be a boolean")


def print_warning(warning):

    print("\033[93mWARNING ("+warning+")\033[0m")