from toxic_classification.service import MLService
from toxic_classification.view import MainInterface


def main():
    service = MLService()
    view = MainInterface(service)
    view.render()


if __name__ == "__main__":
    main()
