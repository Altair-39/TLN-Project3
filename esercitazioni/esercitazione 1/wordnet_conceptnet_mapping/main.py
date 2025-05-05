
from src.wordnet_utils import wordnet_to_conceptnet


def main():
    c = wordnet_to_conceptnet("dog")
    print(c)


if __name__ == "__main__":
    main()
