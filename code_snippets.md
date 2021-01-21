##### How to write argparse using a function and make things moduler

```python
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="test args with function pass",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-hn", "--human_name", default="Soumik", type=str, help="give a name of human"
    )
    parser.add_argument(
        "-num", "--int_number", default=12, type=int, help="provide a integer number"
    )
    return parser.parse_args()


def main():
    args = get_args()
    # you must provide the argparse variable names inside code that has --var_name,
    # not the var_name with single - 
    name = f"My name is {args.human_name}"
    my_favorite_number = f"My favorite number is {args.int_number}"
    print(name)
    print(my_favorite_number)


if __name__ == "__main__":
    main()
```

##### 